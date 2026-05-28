# Распознавание классов зубов человека

## Этапы работы

`1_prepare_dataset.ipynb`

по сырым данным строим датасет с которым можно работать модельке. в dataset_clean сохраненяем все фотографии, на каждом фото ровно один зуб; в файлик `tooth_id_to_class.json` сохраняется маппинг айди фотки - класс. групповые фото разбиваются на ячейки через детекцию сетки линий (проекционный профиль + scipy peaks), каждая ячейка фильтруется CLIP zero-shot классификатором. итого: **3584 фото** 4 классов (incisor / canine / premolar / molar)

`2_model_ready_dataset.ipynb`

строим torch Dataset и Dataloader для работы с нашими данными (деление на train/val/test 70/15/15, аугментации albumentations)

`3_model_v0.ipynb`

бейзлайн (v0) модель классификатора: EfficientNet-B0 с замороженным backbone, дообучается только линейная голова (10 эпох). **Test Accuracy: 82.5%, F1: 0.824**

`3_model_v1.ipynb`

fine-tuning верхних блоков backbone EfficientNet-B0 с дискриминативными learning rates (15 эпох, 3 конфигурации):
- **v1a** — разморозка 1 блока (10.4% параметров): Test Acc **89.2%**, F1 **0.891**
- **v1b** — разморозка 2 блоков (28.3% параметров): Test Acc **91.8%**, F1 **0.919**
- **v1c** — разморозка 3 блоков (78.8% параметров): Test Acc **95.0%**, F1 **0.949** 

`4_compare_models.ipynb`

сравнение всех моделей: метрики, матрицы ошибок, кривые обучения, анализ уверенности

## Деплой модели

`export_to_onnx.py`

конвертация обученной PyTorch-модели в ONNX-формат для запуска в браузере. скрипт загружает чекпоинт `.pth`, экспортирует модель в единый файл `model.onnx` (без внешних данных) для совместимости с ONNX Runtime Web.

```bash
python export_to_onnx.py --checkpoint best_v1c_unfreeze3.pth --out model.onnx
```

`index.html`

веб-интерфейс для классификации зубов прямо в браузере (без сервера). работает полностью offline после загрузки модели (~16 МБ). возможности:
- загрузка одного или нескольких фото (drag & drop или выбор файлов)
- при загрузке нескольких фото — усреднение вероятностей для повышения точности
- индикация уверенности модели: если confidence < 80% — предупреждение с рекомендацией загрузить больше фото
- визуализация вероятностей по всем классам

## Быстрый старт

### Локальное тестирование

```bash
# 1. Установить зависимости
pip install -r requirements.txt

# 2. Сконвертировать модель в ONNX
python export_to_onnx.py

# 3. Запустить локальный сервер
python -m http.server 8080

# 4. Открыть в браузере
# http://localhost:8080
```

### Деплой на GitHub Pages

```bash
# 1. Убедиться что model.onnx и index.html в репозитории
git add model.onnx index.html
git commit -m "add web demo"
git push

# 2. В настройках репозитория: Settings → Pages → Source: main branch / root
# Сайт будет доступен по адресу: https://<username>.github.io/<repo>/
```

---

## Структура проекта

```
├── 1_prepare_dataset.ipynb       # подготовка датасета (CLIP-фильтрация, grid detection)
├── 2_model_ready_dataset.ipynb   # torch Dataset/DataLoader, аугментации
├── 3_model_v0.ipynb              # baseline модель (frozen backbone)
├── 3_model_v1.ipynb              # fine-tuning (3 конфигурации разморозки)
├── 4_compare_models.ipynb        # сравнение всех моделей
├── export_to_onnx.py             # конвертация .pth → .onnx
├── index.html                    # веб-интерфейс (браузерный инференс)
├── model.onnx                    # ONNX-модель для деплоя (16 МБ)
├── requirements.txt              # Python-зависимости
└── utils_*.py                    # вспомогательные модули
```
