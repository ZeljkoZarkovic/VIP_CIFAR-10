#Model za treniranje i poredjenje razlicitih modela.

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import time
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

from model_arhitecture import CIFAR10CNN

#Fiksiranje random seed-ova
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

class ModelComparator:
    #Klasa za poredjenje razlicitih modela

    def __init__(self, classes: List[str]):
        self.classes = classes
        self.results = []

    def train_and_evaluate_model(self, X: np.ndarray, y: np.ndarray, config: Dict, model_name: str, n_folds: int = 5, epochs: int = 50, batch_size: int = 64) -> Dict:
        #Treniranje i evaluacija jednog modela

        print(f"\n{'='*70}")
        print(f"Treniranje: {model_name}")
        print(f"{'='*70}")

        kfold = KFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_SEED)
        
        fold_results = []
        fold_histories = []
        confusion_matrices = []
        training_times = []
        inference_times = []
        
        start_time_total = time.time()

        for fold, (train_idx, val_idx) in enumerate(kfold.split(X), 1):
            print(f"\nFold {fold}/{n_folds}")
            
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            #Kreiranje modela
            cnn = CIFAR10CNN()
            model = cnn.build_custom_model(config)
            cnn.compile_model(learning_rate=config.get('learning_rate', 0.001))

            #CallBacks
            callbacks = [
                keras.callbacks.EarlyStopping(
                    monitor='val_loss', patience=10, restore_best_weights=True
                ),
                keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7
                )
            ]

            #Treniranje
            start_train = time.time()
            history = model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=epochs,
                batch_size=batch_size,
                callbacks=callbacks,
                verbose=0
            )
            train_time = time.time() - start_train
            training_times.append(train_time)

            #Evaluacija
            val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)

            #Inference time
            start_inference = time.time()
            y_pred = model.predict(X_val, verbose=0)
            inference_time = time.time() - start_inference
            inference_times.append(inference_time)

            y_pred_classes = np.argmax(y_pred, axis=1)

            #Confusion matrix
            cm = confusion_matrix(y_val, y_pred_classes)
            confusion_matrices.append(cm)

            #Rezultati
            fold_result = {
                'fold': fold,
                'val_loss': float(val_loss),
                'val_accuracy': float(val_acc),
                'train_loss': float(history.history['loss'][-1]),
                'train_accuracy': float(history.history['accuracy'][-1]),
                'epochs_trained': len(history.history['loss']),
                'train_time': train_time,
                'inference_time': inference_time
            }
            fold_results.append(fold_result)
            fold_histories.append(history.history)
            
            print(f"Val Acc: {val_acc:.4f}, Train Time: {train_time:.2f}s")

        total_time = time.time() - start_time_total

        # Model info
        model_size = sum([np.prod(w.shape) for w in model.get_weights()])
        model_memory = model_size * 4 / (1024**2)  # MB (float32)

        #Agregacija rezultata
        result = {
            'model_name': model_name,
            'config': config,
            'fold_results': fold_results,
            'fold_histories': fold_histories,
            'confusion_matrices': confusion_matrices,
            'avg_val_accuracy': float(np.mean([r['val_accuracy'] for r in fold_results])),
            'std_val_accuracy': float(np.std([r['val_accuracy'] for r in fold_results])),
            'avg_val_loss': float(np.mean([r['val_loss'] for r in fold_results])),
            'avg_train_time': float(np.mean(training_times)),
            'avg_inference_time': float(np.mean(inference_times)),
            'total_time': total_time,
            'model_size_params': int(model_size),
            'model_memory_mb': float(model_memory),
            'timestamp': datetime.now().isoformat()
        }
        
        self.results.append(result)
        
        print(f"\n{model_name} završen!")
        print(f"  Avg Val Acc: {result['avg_val_accuracy']:.4f} ± {result['std_val_accuracy']:.4f}")
        print(f"  Avg Train Time: {result['avg_train_time']:.2f}s")
        print(f"  Model Size: {result['model_size_params']:,} params ({result['model_memory_mb']:.2f} MB)")
        
        return result
    
    def save_results(self, filename: str = 'model_comparison_results.json'):
        #Cuvanje rezultata u JSON
        #Uklonjanje confusion matrices za JSON (prevelike)
        results_to_save = []
        for r in self.results:
            r_copy = r.copy()
            r_copy.pop('confusion_matrices', None)
            r_copy.pop('fold_histories', None)
            results_to_save.append(r_copy)
        
        filepath = Path('logs') / filename
        with open(filepath, 'w') as f:
            json.dump(results_to_save, f, indent=2)
        
        print(f"\nRezultati sačuvani u: {filepath}")

    def create_comparison_table(self) -> pd.DataFrame:
        #Kreiranje tabele za poredenje
        data = []
        for r in self.results:
            data.append({
                'Model': r['model_name'],
                'Val Accuracy': f"{r['avg_val_accuracy']:.4f} ± {r['std_val_accuracy']:.4f}",
                'Val Loss': f"{r['avg_val_loss']:.4f}",
                'Train Time (s)': f"{r['avg_train_time']:.2f}",
                'Inference Time (s)': f"{r['avg_inference_time']:.4f}",
                'Params': f"{r['model_size_params']:,}",
                'Memory (MB)': f"{r['model_memory_mb']:.2f}"
            })
        
        return pd.DataFrame(data)
    
    def plot_comparison(self):
        #Vizuelizacija poredjenja
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle('Model Comparison', fontsize=16, fontweight='bold')
        
        model_names = [r['model_name'] for r in self.results]

        #Validation Accuracy
        val_accs = [r['avg_val_accuracy'] for r in self.results]
        val_stds = [r['std_val_accuracy'] for r in self.results]
        axes[0, 0].bar(model_names, val_accs, yerr=val_stds, capsize=5, color='skyblue', edgecolor='black')
        axes[0, 0].set_ylabel('Validation Accuracy')
        axes[0, 0].set_title('Validation Accuracy')
        axes[0, 0].tick_params(axis='x', rotation=45)
        axes[0, 0].grid(axis='y', alpha=0.3) 

        #Training Time
        train_times = [r['avg_train_time'] for r in self.results]
        axes[0, 1].bar(model_names, train_times, color='coral', edgecolor='black')
        axes[0, 1].set_ylabel('Time (seconds)')
        axes[0, 1].set_title('Average Training Time per Fold')
        axes[0, 1].tick_params(axis='x', rotation=45)
        axes[0, 1].grid(axis='y', alpha=0.3)

        #Model Size
        model_sizes = [r['model_size_params']/1e6 for r in self.results]
        axes[0, 2].bar(model_names, model_sizes, color='lightgreen', edgecolor='black')
        axes[0, 2].set_ylabel('Parameters (millions)')
        axes[0, 2].set_title('Model Size')
        axes[0, 2].tick_params(axis='x', rotation=45)
        axes[0, 2].grid(axis='y', alpha=0.3)

        #Interfence Time
        inf_times = [r['avg_inference_time'] for r in self.results]
        axes[1, 0].bar(model_names, inf_times, color='plum', edgecolor='black')
        axes[1, 0].set_ylabel('Time (seconds)')
        axes[1, 0].set_title('Average Inference Time')
        axes[1, 0].tick_params(axis='x', rotation=45)
        axes[1, 0].grid(axis='y', alpha=0.3)

        #Memory usage
        memory = [r['model_memory_mb'] for r in self.results]
        axes[1, 1].bar(model_names, memory, color='gold', edgecolor='black')
        axes[1, 1].set_ylabel('Memory (MB)')
        axes[1, 1].set_title('Model Memory Usage')
        axes[1, 1].tick_params(axis='x', rotation=45)
        axes[1, 1].grid(axis='y', alpha=0.3)

        #Accuracy vs Time tradeoff
        axes[1, 2].scatter(train_times, val_accs, s=200, alpha=0.6, c=range(len(model_names)), cmap='viridis')
        for i, name in enumerate(model_names):
            axes[1, 2].annotate(name, (train_times[i], val_accs[i]), fontsize=8, ha='center')
        axes[1, 2].set_xlabel('Training Time (s)')
        axes[1, 2].set_ylabel('Validation Accuracy')
        axes[1, 2].set_title('Accuracy vs Training Time')
        axes[1, 2].grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('logs/model_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()

#Definicije 5 razlicitih konfiguracija
MODEL_CONFIGS = {
    'Model_1_Baseline': {
        'num_conv_blocks': 3,
        'filters': [32, 64, 128],
        'kernel_size': 3,
        'dense_units': 256,
        'dropout_rate': 0.5,
        'learning_rate': 0.001
    },
    'Model_2_Deep': {
        'num_conv_blocks': 4,
        'filters': [32, 64, 128, 256],
        'kernel_size': 3,
        'dense_units': 512,
        'dropout_rate': 0.5,
        'learning_rate': 0.001
    },
    'Model_3_Wide': {
        'num_conv_blocks': 3,
        'filters': [64, 128, 256],
        'kernel_size': 3,
        'dense_units': 512,
        'dropout_rate': 0.5,
        'learning_rate': 0.001
    },
    'Model_4_Small': {
        'num_conv_blocks': 2,
        'filters': [32, 64],
        'kernel_size': 3,
        'dense_units': 128,
        'dropout_rate': 0.3,
        'learning_rate': 0.001
    },
    'Model_5_LargeKernel': {
        'num_conv_blocks': 3,
        'filters': [32, 64, 128],
        'kernel_size': 5,
        'dense_units': 256,
        'dropout_rate': 0.5,
        'learning_rate': 0.0005
    }
}