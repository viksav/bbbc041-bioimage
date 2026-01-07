from typing import List, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch
import umap


def plot_confusion_matrix(cm: np.ndarray, labels: Sequence[str], out_path) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_umap(embeddings: np.ndarray, labels: Sequence[int], label_names: Sequence[str], out_path) -> None:
    reducer = umap.UMAP(random_state=42)
    proj = reducer.fit_transform(embeddings)

    fig, ax = plt.subplots(figsize=(6, 5))
    scatter = ax.scatter(proj[:, 0], proj[:, 1], c=labels, cmap="tab10", s=6, alpha=0.8)
    handles, _ = scatter.legend_elements(prop="colors")
    ax.legend(handles, label_names, title="Label", bbox_to_anchor=(1.05, 1), loc="upper left")
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


class GradCAM:
    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module) -> None:
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        self._register_hooks()

    def _register_hooks(self) -> None:
        def forward_hook(_, __, output):
            self.activations = output.detach()

        def backward_hook(_, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def __call__(self, x: torch.Tensor, class_idx: Optional[int] = None) -> np.ndarray:
        self.model.zero_grad(set_to_none=True)
        logits = self.model(x)
        if class_idx is None:
            class_idx = int(torch.argmax(logits, dim=1).item())
        score = logits[:, class_idx].sum()
        score.backward(retain_graph=True)

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = torch.relu(cam)
        cam = torch.nn.functional.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-6)
        return cam


def overlay_heatmap(image: np.ndarray, heatmap: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    heatmap_rgb = plt.cm.jet(heatmap)[:, :, :3]
    overlay = (1 - alpha) * image + alpha * heatmap_rgb
    return np.clip(overlay, 0, 1)


def save_gradcam_grid(
    images: List[np.ndarray],
    heatmaps: List[np.ndarray],
    titles: List[str],
    out_path,
    ncols: int = 4,
) -> None:
    n = len(images)
    ncols = min(ncols, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
    axes = np.atleast_1d(axes).reshape(nrows, ncols)

    for idx in range(nrows * ncols):
        ax = axes.flat[idx]
        if idx >= n:
            ax.axis("off")
            continue
        overlay = overlay_heatmap(images[idx], heatmaps[idx])
        ax.imshow(overlay)
        ax.set_title(titles[idx], fontsize=9)
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
