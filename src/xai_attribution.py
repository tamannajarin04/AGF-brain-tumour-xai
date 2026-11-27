# Copy of the attribution module that the notebooks write to /kaggle/working/xai_attribution.py
# (cell "%%writefile" in 03_agf_main_evaluation.ipynb). Identical in notebooks 03, 04 and 05.
# Kept here for reading and reuse; the notebooks do not import this copy.

"""
xai_attribution.py  — final-1.0 (algorithmically identical to v7.1)
===================================================================
FINAL-VERSION NOTE: no numerical behaviour of any class in this module was
changed relative to the v7.1 run. Only docstrings that contradicted the code
were corrected (the default kernel, and the definition of max-sensitivity).
Brain Tumor MRI Explainability — pruned XAI core.

Provides:
  BaseAttributor, resize_map, _safe_entropy
  GradCAMPlusPlus, SmoothGrad, IntegratedGradients
  SaliencyMetrics (.sensitivity, .gini)
  AgreementGuidedFusion    (AGF — the proposed method)

v7 change (the substantive one):
  0. The agreement kernel is now PLUGGABLE. [CORRECTED DOCSTRING: the default,
     the notebook CONFIG and the executed v7.1 run all use "topk_dice"; an
     earlier draft of this note said "pearson".] The v6 kernel -- 0.5*(SSIM + raw cosine) on
     min-max-normalised NON-NEGATIVE maps -- is bounded well above zero for
     any two maps sharing a coarse spatial prior. Observed score spread was
     ~0.02, so softmax at tau=1 produced a weight spread of ~0.007 and the
     "agreement-guided" fusion was an arithmetic mean to within 2%. That was
     a property of the kernel, not of brain-tumour MRI. "pearson" (centring)
     and "topk_dice" (thresholding) were both implemented; on the synthetic
     shared-prior control only topk_dice produced usable dynamic range, so it
     was fixed as the kernel BEFORE any test-set evaluation. "ssim_cosine" and
     "pearson" are retained as ablation arms.

v6 changes (correctness, not cosmetics):
  1. BaseAttributor._normalize now RAISES on a degenerate (constant) map
     instead of silently returning all zeros. A constant map is never a
     valid attribution: it means the target layer, class index or gradient
     path is broken. The old silent-zero path let a broken GradCAM++ hook
     propagate through AOPC, Gini, entropy and the fusion weights as if it
     were a real explanation.
  2. SmoothGrad gained freeze_noise(): the max-sensitivity metric compares
     phi(x) against phi(x+delta), and with seed_mode="content" those two
     calls hashed to different seeds and drew independent noise sets. That
     made the robustness column measure Monte-Carlo estimator variance on
     top of input sensitivity for the stochastic methods only, while
     GradCAM++ and IG were deterministic. Freezing the draw across the pair
     makes the comparison apples-to-apples.
  3. SaliencyMetrics.sensitivity returns NaN when every perturbation sample
     is non-finite. It previously returned max_rel_diff == 0.0, i.e. it
     scored a total failure as PERFECT robustness.
  4. AgreementGuidedFusion.fuse_maps() fuses PRE-COMPUTED maps, so the
     evaluation loop no longer computes every base method twice (once
     standalone for its own row, once again inside fuse()). It also
     guarantees the standalone maps and the fusion constituents are the
     identical arrays, which the old path did not: fuse() clipped its
     constituents after resize_map and the standalone arm did not, so Gini
     and Sparsity were computed on subtly different data.

REMOVED in v6: two earlier fusion variants that were retired from the
experiment but still occupied ~250 lines of unreachable code. Recover them
from version control if they are ever re-enabled.
"""

from __future__ import annotations
import hashlib
import numpy as np
import torch
from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple, List
import cv2

# SSIM is needed by the ssim_cosine ablation kernel. skimage ships on Kaggle,
# but we fall back to a self-contained windowed-SSIM implementation (scipy
# uniform_filter, standard SSIM formula) so this module still works if
# scikit-image is ever missing -- no silent behaviour change either way.
try:
    from skimage.metrics import structural_similarity as _skimage_ssim
    _HAS_SKIMAGE_SSIM = True
except ImportError:
    _HAS_SKIMAGE_SSIM = False
    from scipy.ndimage import uniform_filter

    def _skimage_ssim(a: np.ndarray, b: np.ndarray, data_range: float = 1.0,
                       win_size: int = 7, **_ignored) -> float:
        """Fallback single-scale SSIM (standard formula, uniform window).
        Only used if scikit-image is unavailable. a, b assumed 2-D float32
        in [0, data_range]."""
        C1, C2 = (0.01 * data_range) ** 2, (0.03 * data_range) ** 2
        a, b = a.astype(np.float64), b.astype(np.float64)
        mu_a, mu_b = uniform_filter(a, win_size), uniform_filter(b, win_size)
        mu_a2, mu_b2, mu_ab = mu_a * mu_a, mu_b * mu_b, mu_a * mu_b
        var_a  = uniform_filter(a * a, win_size) - mu_a2
        var_b  = uniform_filter(b * b, win_size) - mu_b2
        cov_ab = uniform_filter(a * b, win_size) - mu_ab
        num = (2 * mu_ab + C1) * (2 * cov_ab + C2)
        den = (mu_a2 + mu_b2 + C1) * (var_a + var_b + C2)
        return float(np.mean(num / (den + 1e-12)))


# ── Utilities ─────────────────────────────────────────────────────────────

def resize_map(attr_map: np.ndarray, size: Tuple[int, int],
               anti_alias: bool = True) -> np.ndarray:
    h, w = size
    interp = cv2.INTER_LANCZOS4 if anti_alias else cv2.INTER_LINEAR
    return cv2.resize(attr_map.astype(np.float32), (w, h), interpolation=interp)


def _safe_entropy(arr: np.ndarray) -> float:
    if float(arr.max()) < 1e-6:
        return 0.0
    p = arr / (arr.sum() + 1e-8)
    return float(-np.sum(p * np.log(p + 1e-8)))


# ── Abstract base ─────────────────────────────────────────────────────────

class BaseAttributor(ABC):
    def __init__(self, model: torch.nn.Module):
        self.model  = model
        self.device = next(model.parameters()).device

    @abstractmethod
    def generate(self, tensor: torch.Tensor,
                 class_idx: Optional[int] = None) -> Tuple[np.ndarray, int]:
        ...

    def __call__(self, tensor: torch.Tensor,
                 class_idx: Optional[int] = None) -> Tuple[np.ndarray, int]:
        return self.generate(tensor, class_idx)

    @staticmethod
    def _normalize(arr: np.ndarray, *, context: str = "") -> np.ndarray:
        """Min-max normalise an attribution map to [0, 1].

        FAILS LOUDLY on degenerate (constant) input. The previous version
        was `arr -= arr.min(); return arr / (arr.max() + 1e-8)`, which maps
        a constant array to EXACTLY zero and returns it without a warning.
        Every downstream consumer -- aopc_curve, gini, entropy, the
        agreement scores, the weighted fusion sum -- then operates on an
        all-zero map as if it were a real explanation. That single silent
        path is how a mis-resolved GradCAM++ target layer (a 1x1 classifier
        conv) produced a full set of publication tables and a p = 1e-20.

        A constant attribution map is never legitimate. It means one of:
          * the target layer has no spatial extent (classifier head conv),
          * the target layer emits signed activations and GradCAM++'s
            np.maximum(cam, 0) zeroed everything,
          * the class index is wrong, or gradients never reached the hook.
        All of those are bugs, so this raises and lets the caller's
        try/except log a SKIP rather than fabricating a data point.

        `context` is a free-text tag (e.g. "gradcam_pp/vgg16_bn") echoed in
        the error message. It is keyword-only with a default, so existing
        `self._normalize(x)` call sites keep working unchanged.
        """
        arr = np.asarray(arr, dtype=np.float64)
        tag = f" [{context}]" if context else ""
        if not np.isfinite(arr).all():
            n_bad = int(np.count_nonzero(~np.isfinite(arr)))
            raise ValueError(
                f"Non-finite attribution map{tag}: {n_bad} of {arr.size} "
                f"entries are NaN or Inf."
            )
        lo, hi = float(arr.min()), float(arr.max())
        if hi - lo < 1e-12:
            raise ValueError(
                f"Degenerate attribution map{tag}: constant value {lo:.6g} "
                f"(dynamic range {hi - lo:.3g}). Check the target layer "
                f"(spatial extent AND non-negativity), the class index, and "
                f"that gradients reach the hooked module."
            )
        return (arr - lo) / (hi - lo)

    def _resolve_class(self, inp: torch.Tensor,
                       class_idx: Optional[int]) -> int:
        if class_idx is not None:
            return class_idx
        with torch.no_grad():
            return int(self.model(inp).argmax(dim=1).item())


# ── GradCAM++ ─────────────────────────────────────────────────────────────

class GradCAMPlusPlus(BaseAttributor):
    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        super().__init__(model)
        self._acts = self._grads = None
        self._fwd_hook = target_layer.register_forward_hook(self._cache_acts)
        self._bwd_hook = target_layer.register_full_backward_hook(self._cache_grads)

    def _cache_acts(self, _, __, output):
        self._acts = output.detach()

    def _cache_grads(self, _, __, grad_output):
        self._grads = grad_output[0].detach()

    def __enter__(self): return self
    def __exit__(self, *_): self.remove()

    def remove(self):
        self._fwd_hook.remove()
        self._bwd_hook.remove()

    def generate(self, tensor: torch.Tensor,
                 class_idx: Optional[int] = None) -> Tuple[np.ndarray, int]:
        self.model.eval()
        inp       = tensor.unsqueeze(0).to(self.device).requires_grad_(True)
        out       = self.model(inp)
        class_idx = self._resolve_class(inp, class_idx)
        self.model.zero_grad()
        out[0, class_idx].backward()

        grads = self._grads[0].cpu().numpy()
        acts  = self._acts[0].cpu().numpy()
        g2, g3     = grads ** 2, grads ** 3
        spatial_sum = acts.sum(axis=(1, 2), keepdims=True)
        denom  = 2.0 * g2 + spatial_sum * g3
        denom  = np.where(np.abs(denom) > 1e-8, denom, 1e-8)
        alpha  = g2 / denom
        weights = (alpha * np.maximum(grads, 0)).sum(axis=(1, 2))
        cam = np.einsum("c,chw->hw", weights, acts)
        cam = np.maximum(cam, 0)
        return self._normalize(cam, context="gradcam_pp"), class_idx


# ── SmoothGrad ────────────────────────────────────────────────────────────

class SmoothGrad(BaseAttributor):
    """SmoothGrad with reproducible, arm-independent noise.

    seed_mode="content": the generator is reseeded per call from a hash of
    the input tensor's bytes. Consequences that matter for paired testing:
      * the same image draws the same noise whether SmoothGrad is called
        standalone or from inside AGF, so the AGF-vs-SmoothGrad comparison
        is not confounded by a noise mismatch;
      * different images draw different noise, so the fusion arms are not
        reusing one fixed noise set for all N images;
      * results do not depend on call order, so RUN_MODELS staging and
        re-runs reproduce exactly.
    seed_mode="stream": legacy single advancing generator.

    noise_mode="relative": sigma = noise_frac * (input max - min), i.e. the
    fraction-of-range convention SmoothGrad actually specifies.
    noise_mode="absolute": sigma = noise_std, interpreted in whatever space
    the input already lives in (here: ImageNet-normalised).

    NOISE FREEZING (v6) -- required for an honest max-sensitivity metric
    --------------------------------------------------------------------
    Max-sensitivity evaluates  ||phi(x + delta) - phi(x)|| / ||phi(x)||.
    Under seed_mode="content", x and x+delta hash to DIFFERENT seeds, so the
    two calls drew entirely independent noise sets. The measured quantity
    was therefore

        input sensitivity  +  Monte-Carlo estimator variance

    for SmoothGrad and for any fusion containing it, while GradCAM++ and IG
    are deterministic and contributed only the first term. Methods were
    being compared on different measurands.

    freeze_noise(True) caches the noise draw from the next generate() call
    and reuses it for every subsequent call until freeze_noise(False). Wrap
    the (base, perturbed) pair in a freeze and the estimator variance
    cancels, leaving input sensitivity alone. The cache is keyed on tensor
    SHAPE only -- freezing across differently-shaped inputs is a usage
    error and raises rather than silently re-drawing.
    """

    def __init__(self, model: torch.nn.Module, n_samples: int = 50,
                 noise_std: float = 0.10, squared: bool = False,
                 seed: Optional[int] = None,
                 noise_mode: str = "absolute", noise_frac: float = 0.15,
                 seed_mode: str = "content"):
        super().__init__(model)
        self.n_samples  = n_samples
        self.noise_std  = noise_std
        self.squared    = squared
        self.noise_mode = noise_mode
        self.noise_frac = noise_frac
        self.seed_mode  = seed_mode
        self._base_seed = 0 if seed is None else int(seed)
        self._rng = torch.Generator(device="cpu")
        if seed is not None:
            self._rng.manual_seed(self._base_seed)
        # --- noise-freezing state (v6) ---
        self._freeze          = False
        self._frozen_noise    = None      # list[Tensor] on CPU, len == n_samples
        self._frozen_shape    = None
        self._frozen_sigma    = None

    def freeze_noise(self, enable: bool = True) -> "SmoothGrad":
        """Enable/disable noise freezing. Enabling CLEARS any existing cache,
        so the next generate() call defines the frozen draw. Returns self so
        it can be chained: sg.freeze_noise(True).generate(x, c)."""
        self._freeze = bool(enable)
        self._frozen_noise = None
        self._frozen_shape = None
        self._frozen_sigma = None
        return self

    def _content_seed(self, inp: torch.Tensor) -> int:
        """Stable 63-bit seed from the input bytes, mixed with the base seed."""
        buf = inp.detach().to("cpu", torch.float32).contiguous().numpy().tobytes()
        h   = hashlib.blake2b(buf, digest_size=8,
                              person=b"smoothgr").digest()
        return (int.from_bytes(h, "big") ^ (self._base_seed * 0x9E3779B97F4A7C15)) \
               % ((1 << 63) - 1)

    def _draw_noise(self, inp: torch.Tensor, sigma: float) -> List[torch.Tensor]:
        """Return the n_samples noise tensors for this call, honouring the
        freeze cache. Noise is generated on CPU with self._rng (so it is
        reproducible independently of the CUDA RNG) and moved to device."""
        shape = tuple(inp.shape)
        if self._freeze and self._frozen_noise is not None:
            if self._frozen_shape != shape:
                raise ValueError(
                    f"SmoothGrad noise is frozen for shape {self._frozen_shape} "
                    f"but was called with {shape}. Call freeze_noise(True) again "
                    f"to re-arm, or freeze_noise(False) to disable."
                )
            # Reuse the FROZEN sigma as well as the frozen draw. Under
            # noise_mode="relative" sigma is derived from the input's own
            # range, which shifts slightly under perturbation; holding both
            # the draw and its scale fixed is what makes phi(x) and
            # phi(x+delta) differ only by delta.
            return [n.to(inp.device) * self._frozen_sigma
                    for n in self._frozen_noise]
        # Unit-variance draws are cached; the sigma scaling is applied on
        # the way out so the cache stays independent of the noise mode.
        draws = [torch.randn(shape, generator=self._rng)
                 for _ in range(self.n_samples)]
        if self._freeze:
            self._frozen_noise = [d.clone() for d in draws]
            self._frozen_shape = shape
            self._frozen_sigma = float(sigma)
        return [d.to(inp.device) * sigma for d in draws]

    def generate(self, tensor: torch.Tensor,
                 class_idx: Optional[int] = None) -> Tuple[np.ndarray, int]:
        self.model.eval()
        inp       = tensor.unsqueeze(0).to(self.device)
        class_idx = self._resolve_class(inp, class_idx)
        grad_acc  = torch.zeros_like(inp)
        # Reseeding is skipped while frozen: the cached draw already fixes
        # the noise, and reseeding from the PERTURBED input's bytes is
        # exactly the behaviour freezing exists to suppress.
        if self.seed_mode == "content" and not (self._freeze and
                                                self._frozen_noise is not None):
            self._rng.manual_seed(self._content_seed(inp))
        if self.noise_mode == "relative":
            span  = float(inp.max().item() - inp.min().item())
            sigma = self.noise_frac * span
        else:
            sigma = self.noise_std
        for noise in self._draw_noise(inp, sigma):
            noisy = (inp + noise).requires_grad_(True)
            out   = self.model(noisy)
            self.model.zero_grad()
            out[0, class_idx].backward()
            if noisy.grad is not None:
                g = noisy.grad.detach()
                grad_acc += (g ** 2) if self.squared else g.abs()
            del noise, noisy, out
        saliency = grad_acc[0].mean(dim=0).cpu().numpy()
        return self._normalize(saliency, context="smoothgrad"), class_idx

# ── Integrated Gradients ──────────────────────────────────────────────────

class IntegratedGradients(BaseAttributor):
    def __init__(self, model: torch.nn.Module, n_steps: int = 50):
        super().__init__(model)
        self.n_steps = n_steps

    def generate(self, tensor: torch.Tensor,
                 class_idx: Optional[int] = None,
                 baseline: Optional[torch.Tensor] = None) -> Tuple[np.ndarray, int]:
        self.model.eval()
        inp      = tensor.unsqueeze(0).to(self.device)
        baseline = (torch.zeros_like(inp) if baseline is None
                    else baseline.to(self.device))
        class_idx = self._resolve_class(inp, class_idx)
        grad_sum  = torch.zeros_like(inp)
        for k in range(self.n_steps + 1):
            alpha  = k / self.n_steps
            interp = (baseline + alpha * (inp - baseline)).requires_grad_(True)
            out    = self.model(interp)
            self.model.zero_grad()
            out[0, class_idx].backward()
            if interp.grad is not None:
                w = 0.5 if k in (0, self.n_steps) else 1.0
                grad_sum += w * interp.grad.detach()
        ig     = (inp - baseline) * grad_sum / self.n_steps
        ig_map = ig[0].abs().mean(dim=0).cpu().numpy()
        return self._normalize(ig_map, context="ig"), class_idx


# ── SaliencyMetrics ───────────────────────────────────────────────────────

class SaliencyMetrics:
    def __init__(self, model: torch.nn.Module):
        self.model  = model
        self.device = next(model.parameters()).device

    # NOTE (v08): the old SaliencyMetrics.aopc() method was removed. It was
    # dead code (never called; the evaluation loop uses _aopc_curve in section 4,
    # which correctly uses a grey baseline + softmax + insertion/deletion).
    # The old method masked with raw logits and no grey baseline, so keeping it
    # risked being mistaken for the metric actually reported in Table 1.

    def sensitivity(self, tensor: torch.Tensor, attributor,
                    class_idx: int, radius: float = 0.02,
                    n_samples: int = 20, seed: Optional[int] = None,
                    freeze_stochastic: bool = True) -> float:
        """Relative max-sensitivity, adapted from Yeh et al. 2019.

            Sens_rel = max_{s=1..S} ||phi(x+d_s) - phi(x)||_2 / (||phi(x)||_2 + eps),
            d_s ~ N(0, radius^2 I) i.i.d. per element in NORMALISED input space.

        [CORRECTED DOCSTRING] This is a Monte-Carlo maximum over S Gaussian
        perturbations with per-element std `radius`; it is NOT a supremum over
        an L2 ball of radius r (Yeh et al. sample uniformly in an L-inf ball).
        phi(.) is whatever the attributor returns; the notebook wraps every
        method so that phi is the post-processed 224x224 map (min-max ->
        Lanczos resize -> clip >= 0), identical to what the faithfulness
        metrics consume. The class index is fixed by the caller.

        Dividing by the base-map norm makes the score comparable across
        methods whose maps live on different numeric scales; the raw L2
        version ranked methods by MAP MAGNITUDE, not by robustness.

        v6 FIX 1 -- all-NaN case. The loop below only updates max_rel_diff
        on finite samples, and max_rel_diff starts at 0.0. If EVERY sample
        was non-finite the function returned 0.0, i.e. it reported PERFECT
        robustness -- the best possible score -- for a method that failed
        completely on that image. It now returns NaN so the image is
        excluded from the paired comparison instead of silently winning it.

        v6 FIX 2 -- stochastic/deterministic asymmetry. If `attributor`
        exposes freeze_noise() (SmoothGrad, or a fusion that forwards it),
        its noise draw is frozen across the whole (base, perturbed) family.
        Without this, phi(x) and phi(x+d) drew independent SmoothGrad noise
        and the metric measured input sensitivity PLUS Monte-Carlo
        estimator variance -- but only for the stochastic methods, while
        GradCAM++ and IG contributed input sensitivity alone. Set
        freeze_stochastic=False to reproduce the pre-v6 (confounded)
        numbers.

        NOTE: absolute values are not comparable to any pre-v08 run.
        """
        if seed is not None:
            torch.manual_seed(seed)

        # --- freeze stochastic estimators across the whole pair family ----
        _froze = False
        if freeze_stochastic and hasattr(attributor, "freeze_noise"):
            try:
                attributor.freeze_noise(True)
                _froze = True
            except Exception:
                _froze = False
        try:
            base_attr, _ = attributor(tensor, class_idx)
            base_l2      = float(np.linalg.norm(base_attr))
            # A near-zero / degenerate base map drives the relative
            # denominator toward 0 and the ratio explodes (observed pre-fix:
            # ResNet50 GradCAM++ ~8.7e7). Such a map carries no meaningful
            # stability to measure. NOTE: after the v6 target-layer fix this
            # branch should essentially never fire -- if it does, the target
            # layer is still wrong, so it is logged rather than swallowed.
            if base_l2 < 1e-6:
                print("  [WARN] sensitivity(): degenerate base map "
                      f"(L2={base_l2:.3g}) -- image excluded. Check "
                      "get_target_layer / assert_spatial_target.")
                return float("nan")
            base_norm    = base_l2 + 1e-8   # scale reference
            max_rel_diff = 0.0
            n_nan        = 0
            for _ in range(n_samples):
                noise   = torch.randn_like(tensor) * radius
                pert, _ = attributor(tensor + noise, class_idx)
                diff    = float(np.linalg.norm(pert - base_attr)) / base_norm
                # NaN comparisons are always False in Python, so a NaN diff
                # used to silently freeze the max at 0.0 forever.
                if np.isnan(diff):
                    n_nan += 1
                    continue
                if diff > max_rel_diff:
                    max_rel_diff = diff
                del noise, pert
        finally:
            if _froze:
                try:    attributor.freeze_noise(False)
                except Exception: pass

        if n_nan >= n_samples:
            # Every sample failed: returning max_rel_diff (== 0.0) here would
            # score total failure as perfect robustness. Return NaN and let
            # the nan-aware aggregate exclude the image.
            print(f"  [WARN] sensitivity(): all {n_samples} perturbation "
                  "samples were non-finite -- returning NaN (image excluded).")
            return float("nan")
        if n_nan > 0:
            print(f"  [WARN] sensitivity(): {n_nan}/{n_samples} NaN diffs skipped; "
                  f"max taken over the remaining {n_samples - n_nan}.")
        return float(max_rel_diff)

    @staticmethod
    def gini(attr_map: np.ndarray) -> float:
        arr = np.abs(attr_map).flatten().astype(np.float64)
        if arr.sum() < 1e-6:
            return 0.0
        arr = np.sort(arr)
        n   = len(arr)
        idx = np.arange(1, n + 1, dtype=np.float64)
        num = 2.0 * float((idx * arr).sum())
        den = float(n * arr.sum()) + 1e-8
        g   = num / den - (n + 1.0) / n
        return float(max(0.0, g))


# ── ★ AGF — Agreement-Guided Fusion (proposed method) ──────────────────────

# Defaults; Section 2 CONFIG overrides these on the imported module so the
# kernel is set in exactly one place.
_AGF_DEFAULT_KERNEL = "topk_dice"
_AGF_DEFAULT_TAU    = 1.0
_AGF_DEFAULT_TOPK   = 0.10


class AgreementGuidedFusion:
    """
    ★ AGF — Agreement-Guided Fusion.

    Mechanism
    ---------
    For each image, score how much each base method's attribution map agrees
    with the other two, then set per-image fusion weights from those scores.
    A method that agrees with the consensus is trusted more; a method that
    diverges is down-weighted. Weights are recomputed for every image.

    Mathematical definition
    ------------------------
    Let M_1, M_2, M_3 be the resized, non-negative maps for one image and let
    Sim(.,.) be the agreement kernel (below).

    Per-method agreement (mean similarity with the OTHER K-1 methods):
        a_k = (1 / (K-1)) * sum_{j != k} Sim(M_k, M_j)          (K = 3)

    Weights (softmax at temperature tau, tau = 1 in the reported runs):
        w_k = exp(a_k / tau) / sum_l exp(a_l / tau)

    Fused map (weighted arithmetic mean, then re-normalised to [0, 1]):
        AGF(i,j) = sum_k  w_k * M_k(i,j)

    Agreement kernels
    -----------------
    The kernel controls whether the weights can separate at all, so it is a
    first-class, ablatable choice (`agreement_kernel`):

    "topk_dice"  (DEFAULT, proposed)
        Binarise each map at its OWN top-q quantile (q = `topk_frac`) and take
        the Dice coefficient of the two masks, in [0, 1]. Thresholding is the
        operative step: it discards the shared low-level bulk and compares
        only WHERE each method concentrates its evidence. One parameter (q),
        and q is the same quantity a reader already interprets when they look
        at a thresholded saliency overlay.

    "pearson"
        Sim = mean-centred cosine, i.e. the Pearson correlation between the
        two maps, in [-1, 1]. Parameter-free, and it removes the constant
        offset that inflates raw cosine -- but on a synthetic control where
        all three maps carry the SAME coarse spatial prior and differ only in
        fine structure, it scored a spread of 0.022 against ssim_cosine's
        0.023. Centring removes a DC *offset*, not a shared spatial
        *component*, so it does NOT on its own cure the floor effect. Kept as
        an ablation arm because that negative result is worth reporting.

    "ssim_cosine"  (v6 legacy — the ablation control)
        Sim = 0.5 * [SSIM + raw cosine] on min-max-normalised non-negative
        maps. Both terms are bounded well above zero for any two maps sharing
        a coarse spatial prior; the observed score spread on real data was
        ~0.02, and softmax at tau=1 turned that into a weight spread of
        ~0.007, i.e. the fusion was an arithmetic mean to within 2%.

    On the synthetic shared-prior control (three maps, one fine-structure
    outlier) the agreement-score spreads were:
        ssim_cosine 0.023 | pearson 0.022 | topk_dice 0.110
    All three rank the outlier last; they differ in how much dynamic range
    they give the softmax. Section 18 repeats this on real attributions.

    Note that near-uniform weights are NOT automatically a defect: when the
    three base methods genuinely agree, uniform weights are the correct
    answer, and topk_dice returns them (weight std < 0.01 on the all-agree
    control). What was wrong in v6 is that the kernel returned near-uniform
    weights whether the methods agreed or not. Do not claim agreement-guided
    weighting without reporting the weight-spread diagnostic for the kernel
    actually used; Section 18 is that test.

    Returns
    -------
    fuse() returns a 5-tuple:
        (fused_map, resolved_class_idx, individual_maps, weights, agreement_scores)
    fuse_maps() fuses ALREADY-COMPUTED maps and returns a 4-tuple:
        (fused_map, individual_maps, weights, agreement_scores)
    """

    KERNELS = ("topk_dice", "pearson", "ssim_cosine")

    def __init__(self, methods: Dict[str, "BaseAttributor"],
                 ssim_win_size: int = 7,
                 agreement_kernel: Optional[str] = None,
                 temperature: Optional[float] = None,
                 topk_frac: Optional[float] = None,
                 weight_rule: str = "softmax"):
        agreement_kernel = agreement_kernel or _AGF_DEFAULT_KERNEL
        temperature = _AGF_DEFAULT_TAU if temperature is None else temperature
        topk_frac   = _AGF_DEFAULT_TOPK if topk_frac is None else topk_frac
        if agreement_kernel not in self.KERNELS:
            raise ValueError(f"agreement_kernel must be one of {self.KERNELS}, "
                             f"got {agreement_kernel!r}")
        if weight_rule not in ("softmax", "linear"):
            raise ValueError("weight_rule must be 'softmax' or 'linear'")
        if not (0.0 < float(topk_frac) < 1.0):
            raise ValueError("topk_frac must lie strictly in (0, 1)")
        if float(temperature) <= 0.0:
            raise ValueError("temperature must be > 0")
        self.methods          = methods
        self.ssim_win_size    = ssim_win_size
        self.agreement_kernel = agreement_kernel
        self.temperature      = float(temperature)
        self.topk_frac        = float(topk_frac)
        self.weight_rule      = weight_rule

    # ---- similarity primitives ------------------------------------------

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """Raw cosine on the non-negative maps (v6 behaviour)."""
        a_flat = a.flatten().astype(np.float64)
        b_flat = b.flatten().astype(np.float64)
        num = float(np.dot(a_flat, b_flat))
        den = float(np.linalg.norm(a_flat) * np.linalg.norm(b_flat)) + 1e-8
        return num / den

    @staticmethod
    def _pearson_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """Mean-centred cosine == Pearson r, in [-1, 1].

        A constant map (e.g. an identically-zero GradCAM++ on a layer that
        never activates) has zero variance and undefined correlation; it is
        scored 0.0, i.e. 'carries no agreement information', so the softmax
        pushes weight onto the two methods that do agree.
        """
        a_flat = a.flatten().astype(np.float64)
        b_flat = b.flatten().astype(np.float64)
        a_c = a_flat - a_flat.mean()
        b_c = b_flat - b_flat.mean()
        na = float(np.linalg.norm(a_c))
        nb = float(np.linalg.norm(b_c))
        if na < 1e-12 or nb < 1e-12:
            return 0.0
        r = float(np.dot(a_c, b_c) / (na * nb))
        return float(np.clip(r, -1.0, 1.0))

    def _dice_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Dice overlap of the two top-`topk_frac` saliency masks, in [0, 1].

        Each map is thresholded at its OWN quantile, so the two masks always
        have (near-)equal cardinality and Dice is not biased by one method
        being flatter than the other.
        """
        a_flat = a.flatten().astype(np.float64)
        b_flat = b.flatten().astype(np.float64)
        q = 1.0 - self.topk_frac
        ta, tb = np.quantile(a_flat, q), np.quantile(b_flat, q)
        ma, mb = a_flat >= ta, b_flat >= tb
        # A constant map thresholds to the whole image; treat as uninformative.
        if ma.all() or mb.all() or not ma.any() or not mb.any():
            return 0.0
        inter = float(np.logical_and(ma, mb).sum())
        return float(2.0 * inter / (float(ma.sum()) + float(mb.sum()) + 1e-12))

    def _ssim_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        # a, b are already clipped to >= 0 and roughly in [0, 1] (see fuse()).
        a32, b32 = a.astype(np.float32), b.astype(np.float32)
        win = self.ssim_win_size
        # Guard against tiny maps (win_size must not exceed the smaller
        # spatial dimension) -- shrink to the largest valid odd window.
        max_valid = min(a32.shape) if min(a32.shape) % 2 == 1 else min(a32.shape) - 1
        win = min(win, max_valid) if max_valid >= 3 else 3
        try:
            return float(_skimage_ssim(a32, b32, data_range=1.0, win_size=win))
        except Exception:
            # Last-resort fallback: treat as perfectly-dissimilar rather than
            # crashing the whole fusion on a single degenerate image.
            return 0.0

    def kernel_similarity(self, a: np.ndarray, b: np.ndarray,
                          kernel: Optional[str] = None) -> float:
        """Pairwise agreement under `kernel` (defaults to self.agreement_kernel).

        Public so the Section 18 kernel ablation can score one set of
        already-computed maps under every kernel without re-attributing.
        """
        k = kernel or self.agreement_kernel
        if k == "pearson":
            return self._pearson_similarity(a, b)
        if k == "topk_dice":
            return self._dice_similarity(a, b)
        if k == "ssim_cosine":
            return 0.5 * (self._ssim_similarity(a, b) + self._cosine_similarity(a, b))
        raise ValueError(f"unknown agreement kernel {k!r}")

    def _pairwise_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        return self.kernel_similarity(a, b, self.agreement_kernel)

    # ---- agreement + weighting -------------------------------------------

    def agreement_scores(self, maps: Dict[str, np.ndarray],
                         kernel: Optional[str] = None) -> Dict[str, float]:
        """a_k = mean similarity of method k with the other K-1 methods."""
        names = list(maps.keys())
        scores: Dict[str, float] = {}
        for name_i in names:
            sims = [self.kernel_similarity(maps[name_i], maps[name_j], kernel)
                    for name_j in names if name_j != name_i]
            scores[name_i] = float(np.mean(sims))
        return scores

    # back-compat alias for any external caller
    _agreement_scores = agreement_scores

    def weights_from_scores(self, scores: Dict[str, float],
                            temperature: Optional[float] = None,
                            rule: Optional[str] = None) -> Dict[str, float]:
        names = list(scores.keys())
        vals  = np.array([scores[n] for n in names], dtype=np.float64)
        rule  = rule or self.weight_rule
        if rule == "linear":
            c = np.clip(vals, 0.0, None)
            tot = c.sum()
            w = (c / tot) if tot > 1e-12 else np.full(len(c), 1.0 / len(c))
        else:
            tau = float(temperature if temperature is not None else self.temperature)
            v = vals / tau
            v = v - v.max()                       # numerical stability
            e = np.exp(v)
            w = e / (e.sum() + 1e-12)
        return {n: float(wi) for n, wi in zip(names, w)}

    def _softmax_weights(self, scores: Dict[str, float]) -> Dict[str, float]:
        return self.weights_from_scores(scores)

    # ---- fusion over PRE-COMPUTED maps (no attributor calls) -------------

    def fuse_maps(self, indiv: Dict[str, np.ndarray]
                  ) -> Tuple[np.ndarray, Dict[str, np.ndarray],
                             Dict[str, float], Dict[str, float]]:
        """Fuse maps that the caller already computed.

        Added in v6 so the evaluation loop can compute each base method
        EXACTLY ONCE. The old path computed GradCAM++/SmoothGrad/IG
        standalone for their own table rows and then fuse() recomputed all
        three internally -- roughly 52 extra forward/backward passes per
        image, about half the total GPU time of a run.

        It also removes a subtler asymmetry: fuse() clipped its constituents
        to >= 0 after resize_map (Lanczos interpolation can undershoot below
        zero even on a [0,1] input) while the standalone arm did not. Gini,
        Sparsity and Entropy for "GradCAM++" and for "the GradCAM++ inside
        AGF" were therefore computed on slightly different arrays. Routing
        both through this method makes them the same array by construction.

        Returns (fused_map, clipped_individual_maps, weights, agreement).
        """
        if not indiv:
            raise ValueError("fuse_maps() received no maps.")
        indiv = {name: np.clip(np.asarray(m, dtype=np.float32), 0.0, None)
                 for name, m in indiv.items()}
        shapes = {m.shape for m in indiv.values()}
        if len(shapes) != 1:
            raise ValueError(f"fuse_maps() got mismatched map shapes: {shapes}. "
                             "Resize every map to target_size first.")

        agreement = self.agreement_scores(indiv)
        weights   = self.weights_from_scores(agreement)

        fused = np.zeros_like(next(iter(indiv.values())), dtype=np.float64)
        for name, w in weights.items():
            fused += w * indiv[name].astype(np.float64)

        fused = np.clip(fused, 0.0, None)
        fused = fused / (fused.max() + 1e-8)
        fused = np.nan_to_num(fused, nan=0.0, posinf=0.0, neginf=0.0)
        return fused.astype(np.float32), indiv, weights, agreement

    # ---- main entry point --------------------------------------------------

    def fuse(self, tensor, class_idx: Optional[int] = None,
              target_size: Optional[Tuple[int, int]] = None
              ) -> Tuple[np.ndarray, int, Dict[str, np.ndarray],
                         Dict[str, float], Dict[str, float]]:
        """
        Returns (fused_map, resolved_class_idx, individual_maps,
                 weights, agreement_scores).
        """
        indiv: Dict[str, np.ndarray] = {}
        resolved_class = class_idx
        for name, method in self.methods.items():
            attr, pred = method(tensor, resolved_class)
            if resolved_class is None:
                resolved_class = pred
            if target_size is not None:
                attr = resize_map(attr, target_size)
            indiv[name] = attr
        fused, indiv, weights, agreement = self.fuse_maps(indiv)
        return fused, resolved_class, indiv, weights, agreement

    def freeze_noise(self, enable: bool = True) -> "AgreementGuidedFusion":
        """Forward noise freezing to any stochastic constituent.

        Required so SaliencyMetrics.sensitivity() can hold SmoothGrad's noise
        draw fixed across the (base, perturbed) pair when measuring the FUSED
        method, exactly as it does when measuring SmoothGrad standalone.
        Without this the AGF row of the robustness table carries Monte-Carlo
        variance that the GradCAM++ and IG rows do not.
        """
        for m in self.methods.values():
            if hasattr(m, "freeze_noise"):
                m.freeze_noise(enable)
        return self


# Short alias used in the narrative/figures.
AGF = AgreementGuidedFusion
print("xai_attribution final-1.0 (v7.1 numerics) loaded — methods: GradCAM++, SmoothGrad, IG; "
      "proposed fusion: AGF (AgreementGuidedFusion). "
      "kernel=%s; _normalize fails loudly on degenerate maps." % _AGF_DEFAULT_KERNEL)