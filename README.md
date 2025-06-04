# Обработка радиосигналов с использованием нейронных сетей

Проект содержит реализации различных архитектур нейронных сетей для обработки радиосигналов, включая задачи денойзинга и классификации типов модуляции.

## Архитектуры

Проект включает четыре основные модели:
- CNN Denoiser - сверточная нейронная сеть для удаления шума
- CNN Classifier - сверточная нейронная сеть для классификации типов модуляции
- Transformer Denoiser - трансформер для удаления шума
- Transformer Classifier - трансформер для классификации типов модуляции

## Структура проекта

```
├── models/
│   ├── denoisers/
│   │   ├── cnn_denoiser.py
│   │   └── transformer_denoiser.py
│   └── classifiers/
│       ├── cnn_classifier.py
│       └── transformer_classifier.py
├── data/
│   └── full_dataset.npz
├── weights/
└── requirements.txt
```

## Установка

1. Клонируйте репозиторий:
```bash
git clone https://github.com/your-username/radio-signal-processing.git
cd radio-signal-processing
```

2. Установите зависимости:
```bash
pip install -r requirements.txt
```

## Использование

Каждая модель может быть обучена отдельно с помощью соответствующего скрипта:

```bash
python models/denoisers/transformer_denoiser.py
python models/denoisers/cnn_denoiser.py
python models/classifiers/transformer_classifier.py
python models/classifiers/cnn_classifier.py
```

## Особенности реализации

- Реализована ранняя остановка для предотвращения переобучения
- Использован адаптивный learning rate с помощью ReduceLROnPlateau
- Добавлена нормализация данных и BatchNorm
- Реализовано сохранение лучшей модели во время обучения

## Требования

- Python 3.8+
- PyTorch 2.0+
- NumPy
- scikit-learn
- matplotlib
- tqdm

## Лицензия

MIT License 

# Transformer Denoiser

Проект по очистке сигналов от шума с использованием архитектуры Transformer.

## Описание

Данный проект представляет собой реализацию нейронной сети на базе архитектуры Transformer для удаления шума из сигналов. Модель способна обрабатывать двумерные сигналы, эффективно удаляя из них шумовые компоненты.

## Структура проекта

```
├── data/
│   └── full_dataset.npz
├── models/
│   └── denoisers/
│       └── transformer_denoiser.py
├── weights/
│   └── transformer_denoiser.pth
├── requirements.txt
└── README.md
```

## Установка

1. Клонируйте репозиторий:
```bash
git clone https://github.com/GetToNeko/kaloviye-massi.git
cd kaloviye-massi
```

2. Создайте виртуальное окружение и активируйте его:
```bash
python -m venv venv
.\venv\Scripts\activate
```

3. Установите зависимости:
```bash
pip install -r requirements.txt
```

## Использование

1. Подготовьте данные в формате .npz файла и поместите их в директорию `data/`
2. Запустите обучение модели:
```bash
python models/denoisers/transformer_denoiser.py
```

## Особенности модели

- Архитектура: Transformer Encoder
- Размер модели: d_model=128, nhead=4, num_layers=2
- Оптимизатор: Adam
- Функция потерь: MSE
- Scheduler: ReduceLROnPlateau

## Требования

- Python 3.8+
- PyTorch
- NumPy
- scikit-learn
- matplotlib
- tqdm 