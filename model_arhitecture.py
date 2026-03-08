#model_arhitecture.py - Definicija arhitekture neuronske mreze u klasfikaciji CIFAR-10 slika

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from typing import Tuple, Dict
import numpy as np

#Fiksiranje random seed-ova za reproduktivnost
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

class CIFAR10CNN:
    #Konvoluciona neuronska mreža za CIFAR-10 klasifikaciju
    
    def __init__(self, input_shape: Tuple[int, int, int] = (32, 32, 3), 
                 num_classes: int = 10):
        
        #Inicijalizacija CNN modela
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = None

    def build_baseline_model(self) -> keras.Model:
        #Baseline CNN arhitektura

        model = models.Sequential([
            #Conv Block 1
            layers.Conv2D(32, (3, 3), activation='relu', padding='same', 
                         input_shape=self.input_shape),
            layers.BatchNormalization(),
            layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),

            #Conv Block 2
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),

            #Conv Block 3
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),

            #Dense layer
            layers.Flatten(),
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(self.num_classes, activation='softmax')
        ])

        self.model = model
        return model
    
    def build_custom_model(self, config: Dict) -> keras.Model:
        #Prilagodljiva CNN arhitektura sa hiperparametrima

        num_conv_blocks = config.get('num_conv_blocks', 3)
        filters = config.get('filters', [32, 64, 128])
        kernel_size = config.get('kernel_size', 3)
        dense_units = config.get('dense_units', 256)
        dropout_rate = config.get('dropout_rate', 0.5)

        model = models.Sequential()
        model.add(layers.Input(shape=self.input_shape))

        #Konvolucioni blokovi
        for i in range(num_conv_blocks):
            num_filters = filters[i] if i < len(filters) else filters[-1] * (2 ** (i - len(filters) + 1))

            #Dva Conv sloja po bloku
            model.add(layers.Conv2D(num_filters, (kernel_size, kernel_size), 
                                   activation='relu', padding='same'))
            model.add(layers.BatchNormalization())
            model.add(layers.Conv2D(num_filters, (kernel_size, kernel_size), 
                                   activation='relu', padding='same'))
            model.add(layers.BatchNormalization())
            model.add(layers.MaxPooling2D((2, 2)))
            model.add(layers.Dropout(dropout_rate / 2))
        
        #Dense slojevi
        model.add(layers.Flatten())
        model.add(layers.Dense(dense_units, activation='relu'))
        model.add(layers.BatchNormalization())
        model.add(layers.Dropout(dropout_rate))
        model.add(layers.Dense(self.num_classes, activation='softmax'))

        self.model = model
        return model
    
    def compile_model(self, learning_rate: float = 0.001, optimizer: str = 'adam') -> None:
        #Kompajliranje modela
        if optimizer == 'adam':
            opt = keras.optimizers.Adam(learning_rate=learning_rate)
        elif optimizer == 'sgd':
            opt = keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9)
        elif optimizer == 'rmsprop':
            opt = keras.optimizers.RMSprop(learning_rate=learning_rate)
        else:
            opt = keras.optimizers.Adam(learning_rate=learning_rate)

        self.model.compile(
            optimizer=opt,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

    def get_model_summary(self) -> str:
        #Vraca string reprezentaciju arhitekture modela

        if self.model is None:
            return "Model nije kreiran. Pozovite build_baseline_model() ili build_custom_model()."
        
        from io import StringIO
        stream = StringIO()
        self.model.summary(print_fn=lambda x: stream.write(x + '\n'))
        return stream.getvalue()
    
    def count_parameters(self) -> Dict[str, int]:
        #Brojanje parametara modela
        if self.model is None:
            return {"error": "Model nije kreiran"}
        
        trainable_params = sum([tf.size(w).numpy() for w in self.model.trainable_weights])
        non_trainable_params = sum([tf.size(w).numpy() for w in self.model.non_trainable_weights])
        
        return {
            'total_params': trainable_params + non_trainable_params,
            'trainable_params': trainable_params,
            'non_trainable_params': non_trainable_params
        }
    
def create_data_augmentation() -> keras.Sequential:
    #Kreiranje data augmentation pipeline
    return keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
        layers.RandomContrast(0.1),
    ])

def get_callbacks(model_name: str = 'cifar10_model') -> list:
    #Kreiranje callback-a za treniranje
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=f'models/{model_name}_best.h5',
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        )
    ]
    
    return callbacks

if __name__ == "__main__":
    #Test arhitekture
    print("=" * 60)
    print("CIFAR-10 CNN Model Architecture")
    print("=" * 60)
    
    #Baseline model
    print("\n1. Baseline Model:")
    cnn = CIFAR10CNN()
    model = cnn.build_baseline_model()
    cnn.compile_model()
    print(cnn.get_model_summary())
    print(f"\nParametri: {cnn.count_parameters()}")

    #Custom model
    print("\n2. Custom Model:")
    config = {
        'num_conv_blocks': 3,
        'filters': [32, 64, 128],
        'kernel_size': 3,
        'dense_units': 256,
        'dropout_rate': 0.5
    }
    cnn2 = CIFAR10CNN()
    model2 = cnn2.build_custom_model(config)
    cnn2.compile_model(learning_rate=0.001)
    print(cnn2.get_model_summary())
    print(f"\nParametri: {cnn2.count_parameters()}")