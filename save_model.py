#Treniranje i cuvanje svih modela za backend predikciju - Koristi istu konfiguraciju kao u 03_model_comparison.ipynb

import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from PIL import Image
from pathlib import Path
from tqdm import tqdm

#Fiksiranje seed-ova
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck']

#Konfiguracija svih 5 modela
MODEL_CONFIGS = [
    {
        "name": "Model_1_Baseline",
        "num_conv_blocks": 3,
        "filters": [32, 64, 128],
        "kernel_size": 3,
        "dense_units": 256,
        "dropout_rate": 0.5,
        "learning_rate": 0.001,
    },
    {
        "name": "Model_2_Deep",
        "num_conv_blocks": 4,
        "filters": [32, 64, 128, 256],
        "kernel_size": 3,
        "dense_units": 512,
        "dropout_rate": 0.5,
        "learning_rate": 0.001,
    },
    {
        "name": "Model_3_Wide",
        "num_conv_blocks": 3,
        "filters": [64, 128, 256],
        "kernel_size": 3,
        "dense_units": 512,
        "dropout_rate": 0.5,
        "learning_rate": 0.001,
    },
    {
        "name": "Model_4_Small",
        "num_conv_blocks": 2,
        "filters": [32, 64],
        "kernel_size": 3,
        "dense_units": 128,
        "dropout_rate": 0.3,
        "learning_rate": 0.001,
    },
    {
        "name": "Model_5_LargeKernel",
        "num_conv_blocks": 3,
        "filters": [32, 64, 128],
        "kernel_size": 5,
        "dense_units": 256,
        "dropout_rate": 0.5,
        "learning_rate": 0.0005,
    },
]

def load_data(data_dir='data/cifar-10', max_samples=5000):
    print(f"Ucitavanje podataka (max {max_samples} uzoraka)...")
    df = pd.read_csv(Path(data_dir) / 'trainLabels.csv')
    if max_samples:
        df = df.head(max_samples)

    label_to_idx = {label: idx for idx, label in enumerate(CLASSES)}
    train_dir = Path(data_dir) / 'train'

    X, y = [], []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Ucitavanje slika"):
        img_path = train_dir / f"{row['id']}.png"
        if img_path.exists():
            img = np.array(Image.open(img_path), dtype=np.float32) / 255.0
            X.append(img)
            y.append(label_to_idx[row['label']])

    print(f"Ucitano {len(X)} slika")
    return np.array(X), np.array(y, dtype=np.int32)

def build_model(config):
    from model_arhitecture import CIFAR10CNN
    cnn = CIFAR10CNN(input_shape=(32, 32, 3), num_classes=10)
    model = cnn.build_custom_model(config)
    cnn.compile_model(learning_rate=config['learning_rate'])
    return model

def train_and_save(config, X, y):
    name = config['name']
    save_path = f"models/{name}.keras"

    if os.path.exists(save_path):
        print(f"[SKIP] {name} vec postoji: {save_path}")
        return

    print(f"\nTreniranje: {name}")
    model = build_model(config)

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=8, restore_best_weights=True, verbose=0
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=4, min_lr=1e-7, verbose=0
        ),
    ]

    model.fit(
        X, y,
        validation_split=0.2,
        epochs=50,
        batch_size=64,
        callbacks=callbacks,
        verbose=1
    )

    model.save(save_path)
    print(f"Sacuvan: {save_path}")

if __name__ == "__main__":
    os.makedirs('models', exist_ok=True)

    X, y = load_data(max_samples=5000)

    for config in MODEL_CONFIGS:
        train_and_save(config, X, y)

    print("\nSvi modeli su sacuvani u folder 'models/'")