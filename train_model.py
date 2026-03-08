#train_model.py - Treniranje modela sa cross-validacijom, optimiyacijom hiperparametara i logovanjem

import os
import numpy as np
import pandas as pd
import tensorflow as tf
import keras
from sklearn.model_selection import KFold
from PIL import Image
from pathlib import Path
import json
import yaml
from datetime import datetime
from typing import Dict, Tuple, List
import mlflow
import mlflow.keras
import optuna
from tqdm import tqdm

from model_arhitecture import CIFAR10CNN, create_data_augmentation, get_callbacks

#Fiksiranje random seed-ova
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)
os.environ['PYTHONHASHSEED'] = str(RANDOM_SEED)

class CIFAR10Trainer:
    #Klasa za treniranje CIFAR-10 modela

    def __init__(self, data_dir: str = 'data/cifar-10', experiment_name: str = 'cifar10_experiments'):
        #Inicijalizacija trainera
        self.data_dir = Path(data_dir)
        self.train_dir = self.data_dir / 'train'
        self.labels_file = self.data_dir / 'trainLabels.csv'

        self.experiment_name = experiment_name
        self.classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                       'dog', 'frog', 'horse', 'ship', 'truck']
        
        #Kreiranje foldera za modele i logove
        os.makedirs('models', exist_ok=True)
        os.makedirs('logs', exist_ok=True)
        os.makedirs('mlruns', exist_ok=True)

        #MLflow setup
        mlflow.set_experiment(experiment_name)

        #Logovanje verzija biblioteka
        self.log_enviroment()

    def log_enviroment(self):
        #Logovanje verzija biblioteka za reproduktivnost
        env_info = {
            'tensorflow_version': tf.__version__,
            'keras_version': keras.__version__,
            'numpy_version': np.__version__,
            'pandas_version': pd.__version__,
            'python_version': os.sys.version,
            'random_seed': RANDOM_SEED
        }

        with open('logs/environment.json', 'w') as f:
            json.dump(env_info, f, indent=2)

        print("Verzije biblioteka logovane u logs/environment.json")
        return env_info  

    def load_data(self, max_samples: int = None) -> Tuple[np.ndarray, np.ndarray]:
        #Ucitavanje podataka
        print("Učitavanje podataka...")

        #Ucitavanje labela
        df = pd.read_csv(self.labels_file)

        if max_samples:
            df = df.head(max_samples)

        #Mapiranje labela u brojeve
        label_to_idx = {label: idx for idx, label in enumerate(self.classes)}

        X = []
        y = []

        for _, row in tqdm(df.iterrows(), total=len(df), desc="Učitavanje slika"):
            img_path = self.train_dir / f"{row['id']}.png"
            if img_path.exists():
                img = Image.open(img_path)
                img_array = np.array(img)
                X.append(img_array)
                y.append(label_to_idx[row['label']])

        X = np.array(X, dtype=np.float32) / 255.0
        y = np.array(y, dtype=np.int32)

        print(f"  Učitano {len(X)} slika")
        print(f"  Shape: {X.shape}")
        print(f"  Labels: {y.shape}")
        
        return X, y
    
    def train_with_cross_validation(self, X: np.ndarray, y: np.ndarray, config: Dict, n_folds: int = 5, epochs: int = 50, batch_size: int =64) -> Dict:
        #Treniranje sa K-fold cross validacijom

        print(f"\nCross-validation sa {n_folds} fold-ova")
        print("=" * 60)

        kfold = KFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_SEED)
        
        fold_results = []
        fold_histories = []

        with mlflow.start_run(run_name=f"cv_{n_folds}folds_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
            #Logovanje hiperparametara
            mlflow.log_params(config)
            mlflow.log_param('n_folds', n_folds)
            mlflow.log_param('epochs', epochs)
            mlflow.log_param('batch_size', batch_size)
            mlflow.log_param('random_seed', RANDOM_SEED)

            for fold, (train_idx, val_idx) in enumerate(kfold.split(X), 1):
                print(f"\nFold {fold}/{n_folds}")
                print("-" * 60)

                #Split podataka
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]

                print(f"  Train: {len(X_train)} uzoraka")
                print(f"  Val:   {len(X_val)} uzoraka")

                #Kreiranje modela
                cnn = CIFAR10CNN()
                model = cnn.build_custom_model(config)
                cnn.compile_model(learning_rate=config.get('learning_rate', 0.001))

                #Callbacks
                callbacks = [
                    keras.callbacks.EarlyStopping(
                        monitor='val_loss',
                        patience=10,
                        restore_best_weights=True
                    ),
                    keras.callbacks.ReduceLROnPlateau(
                        monitor='val_loss',
                        factor=0.5,
                        patience=5,
                        min_lr=1e-7
                    )
                ]

                #Treniranje
                history = model.fit(
                    X_train, y_train,
                    validation_data=(X_val, y_val),
                    epochs=epochs,
                    batch_size=batch_size,
                    callbacks=callbacks,
                    verbose=0
                )

                #Evaluacija
                val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
                
                print(f"  Val Loss: {val_loss:.4f}")
                print(f"  Val Acc:  {val_acc:.4f}")

                #Cuvanje rezultata
                fold_result = {
                    'fold': fold,
                    'val_loss': float(val_loss),
                    'val_accuracy': float(val_acc),
                    'train_loss': float(history.history['loss'][-1]),
                    'train_accuracy': float(history.history['accuracy'][-1]),
                    'epochs_trained': len(history.history['loss'])
                }
                fold_results.append(fold_result)
                fold_histories.append(history.history)

                #Logovanje metrika po fold-u
                mlflow.log_metrics({
                    f'fold_{fold}_val_loss': val_loss,
                    f'fold_{fold}_val_accuracy': val_acc,
                    f'fold_{fold}_train_loss': history.history['loss'][-1],
                    f'fold_{fold}_train_accuracy': history.history['accuracy'][-1]
                })

                #Logovanje gubitka po epohi
                for epoch, loss in enumerate(history.history['loss']):
                    mlflow.log_metric(f'fold_{fold}_loss_epoch', loss, step=epoch)
                for epoch, val_loss_epoch in enumerate(history.history['val_loss']):
                    mlflow.log_metric(f'fold_{fold}_val_loss_epoch', val_loss_epoch, step=epoch)

            #Agregacija rezultata
            avg_val_loss = np.mean([r['val_loss'] for r in fold_results])
            avg_val_acc = np.mean([r['val_accuracy'] for r in fold_results])
            std_val_acc = np.std([r['val_accuracy'] for r in fold_results])

            print("\n" + "=" * 60)
            print("REZULTATI CROSS-VALIDACIJE")
            print("=" * 60)
            print(f"  Prosečna Val Loss:     {avg_val_loss:.4f}")
            print(f"  Prosečna Val Accuracy: {avg_val_acc:.4f} ± {std_val_acc:.4f}")
            print("=" * 60)

            #Logovanje prosecnih metrika
            mlflow.log_metrics({
                'avg_val_loss': avg_val_loss,
                'avg_val_accuracy': avg_val_acc,
                'std_val_accuracy': std_val_acc
            })

            #Cuvanje rezultata
            results = {
                'config': config,
                'fold_results': fold_results,
                'avg_val_loss': float(avg_val_loss),
                'avg_val_accuracy': float(avg_val_acc),
                'std_val_accuracy': float(std_val_acc),
                'timestamp': datetime.now().isoformat()
            }

            #Cuvanje u JSON
            results_file = f"logs/cv_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2)
            
            mlflow.log_artifact(results_file)
            
            return results
    
    def optimize_hyperparameters(self, X: np.ndarray, y: np.ndarray, n_trials: int = 20, n_folds: int = 3) -> Dict:
        #Optimiyacija hiperparametara sa Optuna

        print(f"\nOptimizacija hiperparametara ({n_trials} trials)")
        print("=" * 60)

        def objective(trial):
            #Definisanje search space-a
            config = {
                'num_conv_blocks': trial.suggest_int('num_conv_blocks', 2, 4),
                'filters': [
                    trial.suggest_categorical('filters_1', [16, 32, 64]),
                    trial.suggest_categorical('filters_2', [32, 64, 128]),
                    trial.suggest_categorical('filters_3', [64, 128, 256])
                ],
                'kernel_size': trial.suggest_categorical('kernel_size', [3, 5]),
                'dense_units': trial.suggest_categorical('dense_units', [128, 256, 512]),
                'dropout_rate': trial.suggest_float('dropout_rate', 0.3, 0.6),
                'learning_rate': trial.suggest_loguniform('learning_rate', 1e-4, 1e-2)
            }

            #Cros-validation
            results = self.train_with_cross_validation(
                X, y, config, n_folds=n_folds, epochs=30, batch_size=64
            )
            
            return results['avg_val_accuracy']
        
        #Optuna study
        study = optuna.create_study(direction='maximize', study_name='cifar10_optimization')
        study.optimize(objective, n_trials=n_trials)

        print("\n" + "=" * 60)
        print("NAJBOLJI HIPERPARAMETRI")
        print("=" * 60)
        print(f"  Best Val Accuracy: {study.best_value:.4f}")
        print(f"  Best Params:")
        for key, value in study.best_params.items():
            print(f"    {key}: {value}")
        print("=" * 60)
        
        #Čuvanje rezultata
        best_params = study.best_params
        best_params['best_value'] = study.best_value
        
        with open('logs/best_hyperparameters.json', 'w') as f:
            json.dump(best_params, f, indent=2)
        
        return best_params
    
def main():
    #Glavna funkcija za treniranje
    print("=" * 60)
    print("CIFAR-10 MODEL TRAINING")
    print("=" * 60)

    #Inicijalizacija
    trainer = CIFAR10Trainer()

    #Ucitavanje podataka (koristimo manji subset za brze testiranje)
    X, y = trainer.load_data(max_samples=5000)

    #Baseline konfiguracija
    baseline_config = {
        'num_conv_blocks': 3,
        'filters': [32, 64, 128],
        'kernel_size': 3,
        'dense_units': 256,
        'dropout_rate': 0.5,
        'learning_rate': 0.001
    }

    #Treniranje sa cross-validacijom
    results = trainer.train_with_cross_validation(
        X, y, baseline_config, n_folds=5, epochs=50, batch_size=64
    )

    print("\nTreniranje završeno!")
    print(f"Rezultati sačuvani u: logs/")
    print(f"MLflow UI: mlflow ui --port 5000")

if __name__ == "__main__":
    main()