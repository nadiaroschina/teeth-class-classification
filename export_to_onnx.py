"""
Export trained EfficientNet-B0 model to ONNX.

Usage:
    python export_to_onnx.py --checkpoint best_v1c_unfreeze3.pth --out model.onnx

Place the resulting model.onnx next to index.html in the repository.

NOTE: weights are inlined into a single file (no .data sidecar) so that
ONNX Runtime Web can load the model in the browser.
"""

import argparse
import tempfile
from pathlib import Path

import onnx
import torch
import torch.nn as nn
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0

NUM_CLASSES = 4
IMG_SIZE = 224


def build_model(num_classes: int) -> nn.Module:
    model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model


def load_checkpoint(model: nn.Module, path: Path) -> nn.Module:
    state = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(state, dict):
        for key in ("model_state_dict", "state_dict"):
            if key in state:
                state = state[key]
                break
    model.load_state_dict(state)
    return model


def export(checkpoint: Path, out: Path) -> None:
    print(f"Loading checkpoint: {checkpoint}")
    model = build_model(NUM_CLASSES)
    model = load_checkpoint(model, checkpoint)
    model.eval()

    dummy = torch.zeros(1, 3, IMG_SIZE, IMG_SIZE)

    # Export to a temp directory so PyTorch does not litter the working dir
    # with .data sidecar files
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir) / "model_tmp.onnx"
        print(f"Exporting to ONNX (tmp): {tmp_path}")
        torch.onnx.export(
            model,
            dummy,
            str(tmp_path),
            input_names=["input"],
            output_names=["logits"],
            dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
            opset_version=17,
        )

        # Load the model together with any external data that may have been created
        print("Loading ONNX model (with external data if any)...")
        onnx_model = onnx.load(str(tmp_path), load_external_data=True)

    # Clear external-data flags on all initializers so weights are stored inline
    for initializer in onnx_model.graph.initializer:
        initializer.ClearField("external_data")
        initializer.data_location = onnx.TensorProto.DEFAULT

    # Save as a single self-contained file
    print(f"Saving inline ONNX: {out}")
    onnx.save(onnx_model, str(out))

    # Remove stale .data file left by a previous run if present
    old_data = Path(str(out) + ".data")
    if old_data.exists():
        old_data.unlink()
        print(f"Removed stale {old_data.name}")

    size_mb = out.stat().st_size / 1024 / 1024
    print(f"\nDone! Size: {size_mb:.1f} MB (single file, no .data sidecar)")
    print(f"Place {out.name} next to index.html and push to the repo.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="best_v1c_unfreeze3.pth", help="Path to .pth model file")
    parser.add_argument("--out", default="model.onnx", help="Output ONNX file path")
    args = parser.parse_args()

    checkpoint = Path(args.checkpoint)
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint.resolve()}")

    export(checkpoint, Path(args.out))
