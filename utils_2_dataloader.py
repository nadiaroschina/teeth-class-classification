import json
import random
from pathlib import Path
from collections import Counter
from typing import Tuple, Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import train_test_split


DATA_DIR = Path("dataset_clean")
IMAGES_DIR = DATA_DIR
JSON_PATH = DATA_DIR / "tooth_id_to_class.json"


CLASS_MAPPING_RULES = {
    'incisor': [f'{x}1' for x in range(1, 5)] + [f'{x}2' for x in range(1, 5)],
    'canine': [f'{x}3' for x in range(1, 5)],
    'premolar': [f'{x}4' for x in range(1, 5)] + [f'{x}5' for x in range(1, 5)],
    'molar': [f'{x}6' for x in range(1, 5)] + [f'{x}7' for x in range(1, 5)] + [f'{x}8' for x in range(1, 5)],
}

# Создаем обратный маппинг: номер зуба -> класс
TOOTH_ID_TO_CLASS_NAME = {}
for class_name, tooth_ids in CLASS_MAPPING_RULES.items():
    for t_id in tooth_ids:
        TOOTH_ID_TO_CLASS_NAME[t_id] = class_name

# Финальный маппинг имени класса в индекс (для потери и вывода модели)
CLASS_NAMES = ['incisor', 'canine', 'premolar', 'molar']
CLASS_NAME_TO_IDX = {name: idx for idx, name in enumerate(CLASS_NAMES)}
IDX_TO_CLASS_NAME = {idx: name for name, idx in CLASS_NAME_TO_IDX.items()}

NUM_CLASSES = len(CLASS_NAMES)

# --- DATASET КЛАСС ---

# Базовый пайплайн для случая, когда transform не передан
_FALLBACK_TRANSFORM = A.Compose([
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])


class ToothDataset(Dataset):
    def __init__(
        self,
        image_paths: List[str],
        labels: List[int],
        transform: Optional[A.Compose] = None,
    ):
        """
        Args:
            image_paths: список полных путей к изображениям.
            labels: список целочисленных меток классов (0 .. NUM_CLASSES-1).
            transform: пайплайн albumentations (должен включать Normalize + ToTensorV2).
                       Если None — применяется минимальный fallback (только нормализация ImageNet).
        """
        assert len(image_paths) == len(labels), (
            f"image_paths и labels должны иметь одинаковую длину: "
            f"{len(image_paths)} != {len(labels)}"
        )
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path = self.image_paths[idx]
        try:
            # PIL.Image.convert("RGB") гарантирует 3-канальный uint8 массив
            image_np = np.array(Image.open(img_path).convert("RGB"))
        except Exception as e:
            raise RuntimeError(f"Не удалось загрузить изображение: {img_path}") from e

        label = self.labels[idx]

        pipeline = self.transform if self.transform is not None else _FALLBACK_TRANSFORM
        image_tensor: torch.Tensor = pipeline(image=image_np)["image"]

        return image_tensor, label

# --- ФУНКЦИИ ПОДГОТОВКИ ДАННЫХ ---

def _build_tooth_mapping(json_path: Path) -> Dict[str, str]:
    """
    Читает JSON вида {index: tooth_id} и возвращает словарь {filename: tooth_id}.

    Поддерживает два формата ключей JSON:
      - числовые индексы (int или строка без паддинга): "0", "1", ...
      - уже готовые имена файлов: "00000.jpg", "00001.jpg", ...
    """
    with open(json_path, "r", encoding="utf-8") as f:
        raw: Dict = json.load(f)

    mapping: Dict[str, str] = {}
    for key, tooth_id in raw.items():
        key_str = str(key).strip()
        # Если ключ уже выглядит как имя файла — используем как есть
        if key_str.endswith(".jpg"):
            fname = key_str
        else:
            # Предполагаем числовой индекс → нормализуем до 5 цифр
            try:
                fname = f"{int(key_str):05d}.jpg"
            except ValueError:
                # Неизвестный формат — пропускаем с предупреждением
                print(f"Warning: не удалось разобрать ключ JSON '{key_str}'. Пропускаем.")
                continue
        mapping[fname] = str(tooth_id)
    return mapping


def load_and_prepare_data(
    data_dir: Path,
    json_path: Path,
    test_size: float = 0.15,
    val_size: float = 0.15,
    seed: int = 42,
) -> Tuple[Tuple, Tuple, Tuple]:
    """
    Загружает пути к файлам, читает JSON, маппит классы и делит на train/val/test.

    Returns:
        (X_train, y_train), (X_val, y_val), (X_test, y_test)
    """
    tooth_mapping = _build_tooth_mapping(json_path)

    # Сортируем для детерминированного порядка
    image_files = sorted(f.name for f in data_dir.glob("*.jpg"))

    all_image_paths: List[str] = []
    all_labels: List[int] = []
    skipped_missing = 0
    skipped_unknown = 0

    for fname in image_files:
        if fname not in tooth_mapping:
            skipped_missing += 1
            continue

        tooth_id_str = tooth_mapping[fname]

        if tooth_id_str not in TOOTH_ID_TO_CLASS_NAME:
            print(f"Warning: неизвестный tooth_id '{tooth_id_str}' для файла {fname}. Пропускаем.")
            skipped_unknown += 1
            continue

        class_name = TOOTH_ID_TO_CLASS_NAME[tooth_id_str]
        all_image_paths.append(str(data_dir / fname))
        all_labels.append(CLASS_NAME_TO_IDX[class_name])

    if skipped_missing:
        print(f"Warning: {skipped_missing} файлов из папки отсутствуют в JSON — пропущены.")
    if skipped_unknown:
        print(f"Warning: {skipped_unknown} файлов с неизвестным tooth_id — пропущены.")

    total = len(all_image_paths)
    print(f"Всего валидных сэмплов: {total}")

    # Распределение по классам
    counts = Counter(all_labels)
    for idx, cnt in sorted(counts.items()):
        print(f"  {IDX_TO_CLASS_NAME[idx]}: {cnt} ({cnt / total * 100:.1f}%)")

    if total == 0:
        raise ValueError("Не найдено ни одного валидного сэмпла. Проверьте DATA_DIR и JSON_PATH.")

    # Стратифицированное разделение: сначала отделяем тест
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        all_image_paths,
        all_labels,
        test_size=test_size,
        random_state=seed,
        stratify=all_labels,
    )

    # Затем отделяем валидацию; пересчитываем долю относительно train_val
    val_ratio = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val,
        y_train_val,
        test_size=val_ratio,
        random_state=seed,
        stratify=y_train_val,
    )

    print(f"Разбивка -> Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)

# --- АУГМЕНТАЦИИ ---

# ImageNet mean/std — стандарт для pretrained моделей
_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD  = [0.229, 0.224, 0.225]


def get_transforms(is_training: bool = True, img_size: int = 224) -> A.Compose:
    """
    Возвращает пайплайн albumentations.

    Для обучения — аугментации + нормализация + ToTensorV2.
    Для валидации/теста — только ресайз + нормализация + ToTensorV2.

    Примечание по GaussNoise: std_range задаётся в нормализованном диапазоне [0, 1].
    Значения (0.039, 0.098) соответствуют умеренному шуму (~10/255 .. 25/255).
    """
    normalize = A.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD)

    if is_training:
        return A.Compose([
            A.Resize(height=img_size, width=img_size),
            A.HorizontalFlip(p=0.5),
            # RandomRotate90 даёт повороты на 0/90/180/270° — агрессивно для зубов,
            # оставляем с низкой вероятностью
            A.RandomRotate90(p=0.2),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
            # std_range в нормализованном диапазоне [0, 1]: 10/255 ≈ 0.039, 25/255 ≈ 0.098
            A.GaussNoise(std_range=(10 / 255, 25 / 255), p=0.3),
            normalize,
            ToTensorV2(),
        ])
    else:
        return A.Compose([
            A.Resize(height=img_size, width=img_size),
            normalize,
            ToTensorV2(),
        ])

# --- СОЗДАНИЕ DATALOADERS ---

def _worker_init_fn(worker_id: int) -> None:
    """Фиксирует seed в каждом worker-процессе для воспроизводимости."""
    seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(seed)
    random.seed(seed)


def create_dataloaders(
    batch_size: int = 32,
    num_workers: int = 4,
    img_size: int = 224,
    data_dir: Path = DATA_DIR,
    json_path: Path = JSON_PATH,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Создаёт DataLoader'ы для train / val / test.

    Returns:
        (train_loader, val_loader, test_loader)
    """
    (train_paths, train_labels), (val_paths, val_labels), (test_paths, test_labels) = (
        load_and_prepare_data(data_dir=data_dir, json_path=json_path, seed=seed)
    )

    train_dataset = ToothDataset(train_paths, train_labels, transform=get_transforms(True,  img_size))
    val_dataset   = ToothDataset(val_paths,   val_labels,   transform=get_transforms(False, img_size))
    test_dataset  = ToothDataset(test_paths,  test_labels,  transform=get_transforms(False, img_size))

    use_pin_memory = torch.cuda.is_available()  # pin_memory not supported on MPS

    common_kwargs = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=use_pin_memory,
        worker_init_fn=_worker_init_fn,
    )

    train_loader = DataLoader(train_dataset, shuffle=True,  **common_kwargs)
    val_loader   = DataLoader(val_dataset,   shuffle=False, **common_kwargs)
    test_loader  = DataLoader(test_dataset,  shuffle=False, **common_kwargs)

    return train_loader, val_loader, test_loader
