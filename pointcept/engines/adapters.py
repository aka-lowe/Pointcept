# Minimal pluggable OTTA adapters
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import pointcept.datasets.transform as T

OTTA_REGISTRY = {}

def register_otta(name):
    def _wrap(cls):
        OTTA_REGISTRY[name] = cls
        return cls
    return _wrap

def build_otta(cfg):
    otta_cfg = getattr(cfg, "otta", {}) or {}
    method = otta_cfg.get("method", "null")
    cls = OTTA_REGISTRY.get(method)
    if cls is None:
        raise KeyError(f"[OTTA] Unknown method: {method}. Available: {list(OTTA_REGISTRY.keys())}")

    print(f"{method} adapter built!")
    return cls(cfg)

class OnlineTTABase:
    def __init__(self, cfg):
        self.cfg = cfg
        self.enabled = bool(getattr(cfg, "otta", {}).get("enabled", False))
        self.steps = int(getattr(cfg, "otta", {}).get("steps", 1))
        self.lr    = float(getattr(cfg, "otta", {}).get("lr", 1e-4))
        self.param_sel = getattr(cfg, "otta", {}).get("params", "bn-affine")
        self._model = None
        self._opt = None

    @property
    def model(self):
        if self._model is None:
            raise RuntimeError("Adapter not attached to model")
        return self._model

    def attach(self, model: torch.nn.Module):
        self._model = model
        if not self.enabled:
            return

        otta_cfg = getattr(self.cfg, "otta", {})
        patterns = otta_cfg.get("params", None)

        # Choose params based on whether we're in logit or feature mode
        if patterns is None:
            if getattr(self, "use_logits", False):
                patterns = ["seg_head."]
            else:
                patterns = ["norm", "projector"]

        params = []
        matched_names = []

        for name, p in model.named_parameters():
            if any(pat in name for pat in patterns):
                p.requires_grad = True
                params.append(p)
                matched_names.append(name)
            else:
                p.requires_grad = False

        if not params:
            raise RuntimeError(
                f"[OTTA] No parameters matched patterns {patterns}. "
                "Try patterns like ['norm', 'projector', 'seg_head.']"
            )

        print(f"[OTTA] Adapting {len(params)} tensors. Examples: {matched_names[:8]}")

        self._opt = torch.optim.SGD(params, lr=self.lr, momentum=0.9)
        self.on_attach()


    # ---- overridables / helpers ----
    def on_attach(self): pass

    def select_params(self, model, which):
        if which == "bn-affine":
            ps = []
            for m in model.modules():
                if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)) and m.affine:
                    if m.weight is not None: ps.append(m.weight)
                    if m.bias   is not None: ps.append(m.bias)
            return ps
        elif which == "none":
            return []
        return []

    def adapt(self, batch):
        if not self.enabled:
            return
        # put only BN into train
        for m in self.model.modules():
            if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                m.train()
        for _ in range(self.steps):
            self._opt.zero_grad(set_to_none=True)
            with torch.enable_grad():
                out = self.model(batch)
                loss = self.compute_loss(out, batch)
                if loss is None:
                    break
                loss.backward()
                self._opt.step()
        self.model.eval()

    def compute_loss(self, outputs, batch):
        return None


@register_otta("null")
class NullAdapter(OnlineTTABase):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.enabled = False
    # compute_loss -> None


@register_otta("entropy")
class EntropyAdapter(OnlineTTABase):
    def compute_loss(self, outputs, batch):
        logits = outputs.get("seg_logits", None)
        if logits is None:
            return None
        p = F.softmax(logits, dim=-1)
        return -(p * (p.clamp_min(1e-12).log())).sum(dim=1).mean()


@register_otta("contrastive")
class ContrastiveAdapter(OnlineTTABase):
    """
    Test-time contrastive adaptation for streaming RGB-D point clouds.
    Uses Pointcept's registered transforms (from config) to create two views.
    """

    # ---------- tiny transform builder ----------
    @staticmethod
    def _build_transforms_from_cfg(cfg_list):
        """Instantiate transforms via the project registry."""
        if cfg_list is None:
            return []
        return [T.TRANSFORMS.build(tcfg) for tcfg in cfg_list]

    # ---------- numpy <-> torch helpers ----------
    @staticmethod
    def _to_numpy_batch(batch):
        out = {}
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                if v.dtype.is_floating_point:
                    out[k] = v.detach().cpu().numpy()
                elif v.dtype in (torch.int32, torch.int64, torch.int16, torch.uint8, torch.bool):
                    out[k] = v.detach().cpu().numpy()
                else:
                    out[k] = v.detach().cpu().numpy()
            else:
                out[k] = v
        return out

    @staticmethod
    def _to_torch_batch(batch_np, ref_batch, device):
        out = {}
        for k, v in batch_np.items():
            rv = ref_batch.get(k, None)
            if isinstance(v, torch.Tensor):
                out[k] = v.to(device)
            elif isinstance(v, (int, float, str)) or v is None:
                out[k] = v
            elif 'coord' in k or 'normal' in k or 'color' in k or isinstance(v, float):
                # float-ish arrays
                t = torch.from_numpy(v).to(device=device, dtype=(rv.dtype if isinstance(rv, torch.Tensor) else torch.float32))
                out[k] = t
            elif isinstance(v, (list, tuple)):
                out[k] = v  # rarely used here
            else:
                # integer-ish arrays (grid_coord, index, inverse, offsets, etc.)
                dtype = torch.long
                if isinstance(rv, torch.Tensor):
                    dtype = rv.dtype
                out[k] = torch.from_numpy(v).to(device=device, dtype=dtype)
        return out

    # ---------- adapter setup ----------
    def on_attach(self):
        c = getattr(self.cfg, "otta", {}).get("contrastive", {}) or {}

        # contrastive knobs
        self.tau = float(c.get("tau", 0.1))
        self.sample_points = int(c.get("sample_points", 4096))
        self.feature_key = c.get("feature_key", None)
        self.use_logits = bool(c.get("use_logits", False))
        self.proj_dim = int(c.get("projector_dim", 128))
        self.detach_backbone = bool(c.get("detach_backbone", False))

        # build Pointcept transforms from config (geometry-only by default)
        default_aug = [
            dict(type="RandomRotate", angle=[-10, 10], axis="z", p=1.0),
            dict(type="RandomScale",  scale=[0.95, 1.05]),
        ]
        self.tta_transforms = self._build_transforms_from_cfg(c.get("augments", default_aug))

        # projector will be lazily created when we see feature dim
        self._projector = None

    # ---------- augmentation application ----------
    def _apply_geo_aug(self, batch):
        """
        Apply project-native transforms safely:
        - convert tensors -> numpy
        - run ptrans transforms
        - convert back -> tensors on original device/dtypes
        """
        device = next(self.model.parameters()).device
        # shallow copy to avoid mutating original batch
        ref_batch = batch
        np_batch = self._to_numpy_batch(ref_batch)
        aug = dict(np_batch)  # copy container

        for t in self.tta_transforms:
            aug = t(aug)

        # Only coords/colors/normals change; still convert robustly
        aug_t = self._to_torch_batch(aug, ref_batch, device)
        return aug_t

    # ---------- projector ----------
    def _ensure_projector(self, in_dim):
        if self._projector is None:
            hidden = max(self.proj_dim, 32)
            self._projector = nn.Sequential(
                nn.Linear(in_dim, hidden, bias=True),
                nn.ReLU(inplace=True),
                nn.Linear(hidden, self.proj_dim, bias=True),
            ).to(next(self.model.parameters()).device)
            # add projector params to optimizer
            if self.enabled and isinstance(self._opt, torch.optim.Optimizer):
                self._opt.add_param_group({"params": self._projector.parameters()})

    # ---------- utils ----------
    def _subsample_idx(self, N):
        dev = next(self.model.parameters()).device
        if self.sample_points <= 0 or self.sample_points >= N:
            return torch.arange(N, device=dev)
        return torch.randperm(N, device=dev)[: self.sample_points]

    def _get_embeddings(self, outputs):
        if self.feature_key is not None:
            emb = outputs.get(self.feature_key, None)
            if emb is None:
                return None
        else:
            emb = outputs.get("feat", None)
            # print(f"DEBUG - Using embeddings")
            if emb is None or self.use_logits:
                # print(f"DEBUG - Using logits")
                emb = outputs.get("seg_logits", None)
        return emb

    def _project_and_norm(self, x):
        self._ensure_projector(x.shape[-1])
        z = self._projector(x)
        return F.normalize(z, dim=-1, eps=1e-6)


    # ---------- main loop ----------
    def adapt(self, batch):
        if not self.enabled:
            return

        # BN in train mode for stats/affine updates
        for m in self.model.modules():
            if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                m.train()

        # two views
        v1 = self._apply_geo_aug(batch)
        v2 = self._apply_geo_aug(batch)

        for _ in range(self.steps):
            self._opt.zero_grad(set_to_none=True)
            with torch.enable_grad():
                out1 = self.model(v1)
                out2 = self.model(v2)

                f1 = self._get_embeddings(out1)
                f2 = self._get_embeddings(out2)
                if f1 is None or f2 is None:
                    break

                if self.detach_backbone:
                    f1 = f1.detach()
                    f2 = f2.detach()

                M = min(f1.shape[0], f2.shape[0])
                idx = self._subsample_idx(M)
                z1 = self._project_and_norm(f1[idx])
                z2 = self._project_and_norm(f2[idx])

                logits = (z1 @ z2.t()) / self.tau
                targets = torch.arange(logits.shape[0], device=logits.device)
                loss = F.cross_entropy(logits, targets)
                
                print(f"DEBUG - LOSS: {loss}")

                loss.backward()
                self._opt.step()

        self.model.eval()

    
@register_otta("tent")
class TentAdapter(OnlineTTABase):
    """
    TENT-LN: entropy minimization on test batches
    for transformer-style backbones with LayerNorm/GroupNorm.
    Updates ONLY norm affine (weight/bias). No running stats involved.
    """

    def on_attach(self):
        # 1) Collect LN/GN affine params and their NAMES
        adapt_params = []
        adapt_names  = set()
        for module_name, m in self.model.named_modules():
            if isinstance(m, (nn.LayerNorm, nn.GroupNorm)):
                if getattr(m, "weight", None) is not None:
                    m.weight.requires_grad = True
                    adapt_params.append(m.weight)
                    adapt_names.add(f"{module_name}.weight")
                if getattr(m, "bias", None) is not None:
                    m.bias.requires_grad = True
                    adapt_params.append(m.bias)
                    adapt_names.add(f"{module_name}.bias")

        if not adapt_params:
            raise RuntimeError("[TENT] No LayerNorm/GroupNorm affine parameters found.")

        # 2) Freeze everything else BY NAME (avoid tensor equality)
        for name, p in self.model.named_parameters():
            p.requires_grad = (name in adapt_names)

        # 3) Optimizer over just the LN/GN affine params
        self._opt = torch.optim.SGD(adapt_params, lr=self.lr, momentum=0.9)

        # 4) Optional episodic snapshot (store source LN/GN affine only)
        self._snapshot = {
            name: p.detach().clone()
            for name, p in self.model.named_parameters()
            if name in adapt_names
        }

        print(f"[TENT-LN] Adapting {len(adapt_params)} tensors. "
            f"Examples: {list(sorted(adapt_names))[:8]}")


    @staticmethod
    def _entropy_from_outputs(outputs: dict):
        logits = outputs.get("seg_logits", None)
        if logits is None:
            return None
        p = F.softmax(logits, dim=-1)
        return -(p * (p.clamp_min(1e-12).log())).sum(dim=1).mean()

    def adapt(self, batch):
        if not self.enabled:
            return

        # Use eval() so dropout/etc. are disabled; LN/GN don’t care about train/eval.
        self.model.eval()

        for _ in range(self.steps):
            self._opt.zero_grad(set_to_none=True)
            with torch.enable_grad():
                out = self.model(batch)   # forward on the ORIGINAL batch (no aug)
                loss = self._entropy_from_outputs(out)
                print(f"DEBUG - Loss: {loss}")
                if loss is None:
                    break
                loss.backward()
                self._opt.step()

        # stay in eval() for the subsequent prediction in your tester
        self.model.eval()
