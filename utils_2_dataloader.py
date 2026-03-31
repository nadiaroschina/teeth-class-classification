import os
import json
from pathlib import Path
import pandas as pd
import numpy as np

from typing import Tuple, Dict, List, Optional

import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import train_test_split


DATA_DIR = Path("dataset_clean_v2")
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

class ToothDataset(Dataset):
    def __init__(self, image_paths: List[str], labels: List[int], transform: Optional[A.Compose] = None):
        """
        image_paths: список полных путей к картинкам
        labels: список целочисленных меток классов (0-3)
        transform: альбументации трансформации
        """
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        # 1. Загрузка изображения (PIL открывает как RGB автоматически)
        img_path = self.image_paths[idx]
        try:
            image = Image.open(img_path).convert("RGB")
            image_np = np.array(image)
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Возвращаем заглушку или пропускаем (в реальном проекте лучше логировать и исключать)
            raise e

        label = self.labels[idx]

        # 2. Применение аугментаций
        if self.transform:
            # Albumentations ожидает словарь {'image': np_array}
            transformed = self.transform(image=image_np)
            image_tensor = transformed['image']
        else:
            # Базовое преобразование в тензор, если аугментаций нет (для теста)
            transform_basic = A.Compose([A.Normalize(), ToTensorV2()])
            image_tensor = transform_basic(image=image_np)['image']

        return image_tensor, label

# --- ФУНКЦИИ ПОДГОТОВКИ ДАННЫХ ---

def load_and_prepare_data(data_dir: Path, json_path: Path, test_size: float = 0.15, val_size: float = 0.15, seed: int = 42):
    """
    Загружает пути к файлам, читает JSON, маппит классы и делит на train/val/test.
    Возвращает списки путей и меток для каждого сплита.
    """
    
    # 1. Чтение маппинга
    with open(json_path, 'r', encoding='utf-8') as f:
        tooth_mapping_raw = json.load(f)

    tooth_mapping = {
        f'{str(i).zfill(5)}.jpg': tooth_id for i, tooth_id in tooth_mapping_raw.items()
    }
    
    # tooth_mapping ожидается как {"00000.jpg": "11", "00001.jpg": "24", ...}
    # Убедимся, что ключи совпадают с именами файлов
    
    # 2. Сбор всех валидных данных
    all_image_paths = []
    all_labels = []
    
    # Получаем список всех jpg файлов
    # Сортируем, чтобы порядок был детерминированным
    image_files = sorted([f.name for f in data_dir.glob("*.jpg")])
    
    missing_count = 0
    
    for fname in image_files:
        if fname not in tooth_mapping:
            # Если файла нет в JSON, пропускаем (или можно выбросить ошибку)
            missing_count += 1
            continue
            
        tooth_id_str = tooth_mapping[fname]
        
        # Нормализуем ключ: иногда в JSON может быть "11", а нужно сравнить со строкой
        # В нашем маппинге ключи строки ('11', '12'...)
        
        if tooth_id_str not in TOOTH_ID_TO_CLASS_NAME:
            print(f"Warning: Unknown tooth ID {tooth_id_str} for file {fname}. Skipping.")
            continue
            
        class_name = TOOTH_ID_TO_CLASS_NAME[tooth_id_str]
        class_idx = CLASS_NAME_TO_IDX[class_name]
        
        full_path = str(data_dir / fname)
        all_image_paths.append(full_path)
        all_labels.append(class_idx)
        
    if missing_count > 0:
        print(f"Warning: {missing_count} images found in folder but not in JSON were skipped.")
        
    print(f"Total valid samples loaded: {len(all_image_paths)}")
    
    # 3. Стратифицированное разделение
    # Сначала отделяем тест
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        all_image_paths, all_labels, 
        test_size=test_size, 
        random_state=seed, 
        stratify=all_labels # Важно для баланса классов
    )
    
    # Затем отделяем валидацию от тренировочной части
    # Вычисляем долю валидации относительно оставшейся части (train_val)
    # Если мы хотим 15% от всего датасета на вал, то доля от остатка будет: 0.15 / (1 - 0.15)
    val_ratio = val_size / (1 - test_size)
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val,
        test_size=val_ratio,
        random_state=seed,
        stratify=y_train_val
    )
    
    print(f"Split sizes -> Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)

# --- АУГМЕНТАЦИИ ---

def get_transforms(is_training: bool = True, img_size: int = 224):
    """
    Возвращает пайплайн аугментаций.
    Для обучения: сильные аугментации.
    Для валидации/теста: только ресайз и нормализация.
    """
    if is_training:
        return A.Compose([
            A.Resize(height=img_size, width=img_size),
            A.HorizontalFlip(p=0.5), # Осторожно: допустимо ли для вашей задачи? Для типа зуба - обычно да.
            A.RandomRotate90(p=0.2), # Небольшие повороты
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
            A.GaussNoise(std_range=[0.1, 0.2], p=0.3),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), # Stats от ImageNet
            ToTensorV2()
        ])
    else:
        return A.Compose([
            A.Resize(height=img_size, width=img_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])

# --- СОЗДАНИЕ DATALOADERS ---

def create_dataloaders(batch_size: int = 32, num_workers: int = 4, img_size: int = 224, data_dir=DATA_DIR, json_path=JSON_PATH):
    # 1. Подготовка списков данных
    (train_paths, train_labels), (val_paths, val_labels), (test_paths, test_labels) = load_and_prepare_data(
        data_dir=data_dir, 
        json_path=json_path
    )

    print(f"Train: {len(train_paths)}, Val: {len(val_paths)}, Test: {len(test_paths)}")
    
    # 2. Создание объектов Dataset
    train_dataset = ToothDataset(train_paths, train_labels, transform=get_transforms(is_training=True, img_size=img_size))
    val_dataset = ToothDataset(val_paths, val_labels, transform=get_transforms(is_training=False, img_size=img_size))
    test_dataset = ToothDataset(test_paths, test_labels, transform=get_transforms(is_training=False, img_size=img_size))
    
    # 3. Создание DataLoader
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, # Перемешивание важно для обучения
        num_workers=num_workers,
        pin_memory=(True if torch.cuda.is_available() else False) # Ускоряет передачу на GPU
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=(True if torch.cuda.is_available() else False)
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=(True if torch.cuda.is_available() else False)
    )
    
    return train_loader, val_loader, test_loader
