"""
Конвертация обученной модели EfficientNet-B0 → ONNX (один файл).

Запуск:
    python export_to_onnx.py
    python export_to_onnx.py --checkpoint best_v1a.pth --out model.onnx
"""

import argparse
import os
from pathlib import Path

import torch
import torch.nn as nn
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0

NUM_CLASSES = 4
IMG_SIZE    = 224


def build_model(num_classes: int) -> nn.Module:
    model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model


def load_checkpoint(model: nn.Module, path: Path) -> nn.Module:
    state = torch.load(path, map_location="cpu")
    if isinstance(state, dict):
        for key in ("model_state_dict", "state_dict"):
            if key in state:
                state = state[key]
                break
    model.load_state_dict(state)
    return model


def export(checkpoint: Path, out: Path) -> None:
    print(f"Загружаю чекпоинт: {checkpoint}")
    model = build_model(NUM_CLASSES)
    model = load_checkpoint(model, checkpoint)
    model.eval()

    dummy = torch.zeros(1, 3, IMG_SIZE, IMG_SIZE)

    # Удаляем старые файлы если есть
    for f in [out, Path(str(out) + ".data")]:
        if f.exists():
            os.remove(f)
            print(f"Удалён старый файл: {f.name}")

    print(f"Экспортирую в ONNX: {out}")
    torch.onnx.export(
        model,
        dummy,
        str(out),
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=11,          # 11 максимально совместим с onnxruntime-web
        do_constant_folding=True,
    )

    # Проверяем что .data не создался
    data_file = Path(str(out) + ".data")
    if data_file.exists():
        raise RuntimeError(
            f"Всё равно создался {data_file.name}.\n"
            "Попробуйте: pip install --upgrade torch torchvision"
        )

    size_mb = out.stat().st_size / 1024 / 1024
    print(f"\nГотово! Один файл: {out.name} ({size_mb:.1f} МБ)")
    print("Удалите старые model.onnx и model.onnx.data из репозитория,")
    print("добавьте новый model.onnx и запушьте.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="best_v1c.pth")
    parser.add_argument("--out", default="model.onnx")
    args = parser.parse_args()

    checkpoint = Path(args.checkpoint)
    if not checkpoint.exists():
        raise FileNotFoundError(f"Чекпоинт не найден: {checkpoint.resolve()}")

    export(checkpoint, Path(args.out))
