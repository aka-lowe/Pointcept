from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_scatter

from timm.layers import trunc_normal_
from pointcept.models.builder import MODELS, build_model
from pointcept.models.modules import PointModel
from pointcept.models.utils.structure import Point
from pointcept.utils.scheduler import CosineScheduler
from pointcept.utils.comm import all_gather, get_world_size


def l2norm(x):
    return F.normalize(x, dim=-1)


def batch_mean(feat, batch):
    return torch_scatter.segment_coo(feat, batch, reduce="mean")


class LazyProjHead(nn.Module):
    """Projection head that learns input dim at first forward (no need to know backbone dims)."""

    def __init__(self, out_dim=512, hidden=1024, bn=False, drop=0.0):
        super().__init__()
        self.fc1 = nn.LazyLinear(hidden)
        self.bn = nn.BatchNorm1d(hidden) if bn else nn.Identity()
        self.act = nn.GELU()
        self.drop = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden, out_dim)
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        return l2norm(x)


class TranslatorMLP(nn.Module):
    """Ego -> Exo-like feature mapper (gated residual)."""

    def __init__(self, dim=None, hidden=None, layers=2, dropout=0.0):
        super().__init__()
        # Use LazyLinear for auto input dim
        self.ln = nn.LayerNorm(dim if dim is not None else 1)
        self.use_lazy_ln = dim is None
        blocks = []
        for i in range(layers):
            blocks += [
                nn.LazyLinear(hidden or 1024)
                if i == 0 and dim is None
                else nn.Linear(dim if i == 0 else (hidden or 1024), (hidden or 1024)),
                nn.GELU(),
                nn.Dropout(dropout),
            ]
        self.mlp = nn.Sequential(*blocks, nn.LazyLinear(dim or 1))
        self.gate = nn.Sequential(nn.LazyLinear(dim or 1), nn.Sigmoid())
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, z_ego):
        if self.use_lazy_ln and isinstance(self.ln.normalized_shape, int):
            # Rebuild LayerNorm with correct dim on first call
            self.ln = nn.LayerNorm(z_ego.shape[-1]).to(z_ego.device)
            self.use_lazy_ln = False
        z = self.mlp(self.ln(z_ego))
        g = self.gate(z_ego)
        z_hat_exo = z_ego + g * z
        return l2norm(z_hat_exo)


class ProtoAssign(nn.Module):
    """Soft prototype distribution for soft weights (auto-dim)."""

    def __init__(self, hidden=4096, embed=512, num_proto=4096, temp=0.07):
        super().__init__()
        self.m1 = nn.LazyLinear(hidden)
        self.act = nn.GELU()
        self.m2 = nn.Linear(hidden, embed)
        self.proto = nn.Linear(embed, num_proto, bias=False)
        self.temp = temp
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, z):
        e = l2norm(self.m2(self.act(self.m1(z))))
        logits = self.proto(e)
        return F.softmax(logits / self.temp, dim=-1)


@MODELS.register_module("Sonata-Translated")
class SonataTranslated(PointModel):
    """
    Dual-encoder + MLP translator training (no text). Auto-detects dims; works with paired or unpaired batches.

    Expected in data_dict:
      ego_feat, ego_coord, ego_origin_coord, ego_offset, grid_size
      (optional) exo_feat, exo_coord, exo_origin_coord, exo_offset
    """

    def __init__(
        self,
        backbone,
        proj_out_dim=512,
        proj_hidden=1024,
        translator_hidden=None,
        translator_layers=2,
        translator_dropout=0.0,
        share_backbones=True,
        temperature=0.07,
        lambda_vv=1.0,
        lambda_vs=0.5,
        lambda_vs_warmup_ratio=0.1,
        queue_size=32768,
        up_cast_level=0,
        proto_hidden=4096,
        proto_embed=512,
        proto_num=4096,
        proto_temp=0.07,
    ):
        super().__init__()
        self.up_cast_level = up_cast_level
        # Encoders
        self.ego_backbone = build_model(dict(backbone))
        self.exo_backbone = self.ego_backbone if share_backbones else build_model(dict(backbone))
        # Projectors (auto-dim)
        self.ego_head = LazyProjHead(out_dim=proj_out_dim, hidden=proj_hidden)
        self.exo_head = LazyProjHead(out_dim=proj_out_dim, hidden=proj_hidden)
        # Translator & Prototypes (auto-dim)
        self.translator = TranslatorMLP(
            dim=None,
            hidden=translator_hidden,
            layers=translator_layers,
            dropout=translator_dropout,
        )
        self.proto = ProtoAssign(
            hidden=proto_hidden,
            embed=proto_embed,
            num_proto=proto_num,
            temp=proto_temp,
        )
        # Loss weights & temp
        self.tau = temperature
        self.lambda_vv = float(lambda_vv)
        self.lambda_vs = float(lambda_vs)
        self.lambda_vs_base = float(lambda_vs)
        self.lambda_vs_warmup_ratio = lambda_vs_warmup_ratio
        # Queues
        self.queue_size = queue_size
        self.register_buffer("queue_exo_hat", l2norm(torch.randn(queue_size, proj_out_dim)))
        self.register_buffer("queue_exo", l2norm(torch.randn(queue_size, proj_out_dim)))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
        # Scheduler for λ_vs
        self.lambda_vs_scheduler = None

    # --- optional: up-cast like Sonata if your backbone emits pooling_* ---
    def _up_cast(self, point):
        for _ in range(self.up_cast_level):
            assert "pooling_parent" in point.keys() and "pooling_inverse" in point.keys()
            parent = point.pop("pooling_parent")
            inverse = point.pop("pooling_inverse")
            parent.feat = torch.cat([parent.feat, point.feat[inverse]], dim=-1)
            point = parent
        return point

    def _encode_clip(self, pc_dict, backbone):
        point = Point(pc_dict)
        point = backbone(point)
        if self.up_cast_level > 0:
            point = self._up_cast(point)
        z_raw = batch_mean(point.feat, point.batch)
        return z_raw

    def before_train(self):
        total_steps = self.trainer.cfg.scheduler.total_steps
        curr = self.trainer.start_epoch * len(self.trainer.train_loader)
        self.lambda_vs_scheduler = CosineScheduler(
            start_value=0.0,
            base_value=self.lambda_vs_base,
            final_value=self.lambda_vs_base,
            warmup_iters=int(total_steps * self.lambda_vs_warmup_ratio),
            total_iters=total_steps,
        )
        self.lambda_vs_scheduler.iter = curr

    def before_step(self):
        self.lambda_vs = self.lambda_vs_scheduler.step()
        if self.trainer.writer is not None:
            self.trainer.writer.add_scalar(
                "params/lambda_vs", self.lambda_vs, self.lambda_vs_scheduler.iter
            )

    @torch.no_grad()
    def _dequeue_and_enqueue(self, z_exo_hat, z_exo):
        n = z_exo_hat.size(0)
        ptr = int(self.queue_ptr.item())
        end = ptr + n

        def place(buf, src):
            if src is None:
                return
            if end <= self.queue_size:
                buf[ptr:end] = src.detach()
            else:
                k = self.queue_size - ptr
                buf[ptr:] = src[:k].detach()
                buf[: n - k] = src[k:].detach()

        place(self.queue_exo_hat, z_exo_hat)
        place(self.queue_exo, z_exo if z_exo is not None else None)
        self.queue_ptr[0] = (ptr + n) % self.queue_size

    @staticmethod
    def _info_nce(q, k, neg, tau):
        pos = (q * k).sum(dim=-1, keepdim=True)
        logits = torch.cat([pos, q @ neg.t()], dim=1)
        labels = torch.zeros(q.size(0), dtype=torch.long, device=q.device)
        return F.cross_entropy(logits.float() / tau, labels)

    def forward(self, data_dict):
        ego_pc = dict(
            feat=data_dict["ego_feat"],
            coord=data_dict["ego_coord"],
            origin_coord=data_dict["ego_origin_coord"],
            offset=data_dict["ego_offset"],
            grid_size=data_dict["grid_size"][0],
        )
        z_ego_raw = self._encode_clip(ego_pc, self.ego_backbone)
        z_ego = self.ego_head(z_ego_raw)

        paired = "exo_feat" in data_dict
        if paired:
            exo_pc = dict(
                feat=data_dict["exo_feat"],
                coord=data_dict["exo_coord"],
                origin_coord=data_dict["exo_origin_coord"],
                offset=data_dict["exo_offset"],
                grid_size=data_dict["grid_size"][0],
            )
            z_exo_raw = self._encode_clip(exo_pc, self.exo_backbone)
            z_exo = self.exo_head(z_exo_raw)
        else:
            z_exo_raw, z_exo = None, None

        z_exo_hat = self.translator(z_ego.detach())

        z_exo_hat_all = all_gather(z_exo_hat)
        neg_exo_hat = torch.cat([z_exo_hat_all, self.queue_exo_hat], dim=0)
        if z_exo is not None and z_exo.numel() > 0:
            z_exo_all = all_gather(z_exo)
            neg_exo = torch.cat([z_exo_all, self.queue_exo], dim=0)
        else:
            neg_exo = torch.cat([all_gather(z_ego), self.queue_exo_hat], dim=0)

        loss_vs = self._info_nce(z_ego, z_exo_hat, neg_exo_hat, self.tau)

        if z_exo is not None and z_exo.numel() > 0:
            sim_e2x = z_exo_hat @ z_exo.t()
            idx_e2x = sim_e2x.argmax(dim=1)
            z_exo_pos = z_exo[idx_e2x]

            with torch.no_grad():
                ego_hat_from_exo = self.translator(z_exo.detach())
                sim_x2e = ego_hat_from_exo @ z_ego.t()
                idx_x2e = sim_x2e.argmax(dim=1)
                mnn = (
                    torch.arange(z_ego.size(0), device=z_ego.device)
                    == idx_x2e[idx_e2x]
                ).float()

            with torch.no_grad():
                p_ego = self.proto(z_ego)
                p_exo = self.proto(z_exo_pos)
                proto_sim = (p_ego * p_exo).sum(-1).clamp(0, 1)
                w = proto_sim * mnn

            pos = (z_ego * z_exo_pos).sum(-1, keepdim=True)
            logits = torch.cat([pos, z_ego @ neg_exo.t()], dim=1)
            labels = torch.zeros(z_ego.size(0), dtype=torch.long, device=z_ego.device)
            ce = F.cross_entropy(logits.float() / self.tau, labels, reduction="none")
            loss_vv = (w * ce).sum() / (w.sum() + 1e-6)
        else:
            loss_vv = torch.tensor(0.0, device=z_ego.device)

        loss = self.lambda_vs * loss_vs + self.lambda_vv * loss_vv

        with torch.no_grad():
            self._dequeue_and_enqueue(
                z_exo_hat,
                z_exo if (z_exo is not None and z_exo.numel() > 0) else None,
            )

        if get_world_size() > 1:
            for t in [loss, loss_vs, loss_vv]:
                torch.distributed.all_reduce(t, op=torch.distributed.ReduceOp.AVG)

        return {"loss": loss, "loss_vs": loss_vs, "loss_vv": loss_vv}
