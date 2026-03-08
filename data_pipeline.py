import os
import pandas as pd
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Tuple, Dict, List
import warnings
warnings.filterwarnings('ignore')

class CIFAR10DataPipeline:
    def __init__(self, data_dir: str = 'data/cifar-10'):
        self.data_dir = Path(data_dir)
        self.train_dir = self.data_dir / 'train'
        self.test_dir = self.data_dir / 'test'
        self.labels_file = self.data_dir / 'trainLabels.csv'

        #CIFAR-10 klase
        self.classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                        'dog', 'frog', 'horse', 'ship', 'truck']
        self.train_df = None
        self.validation_results = {}

    def load_labels(self) -> pd.DataFrame:
        print("Ucitavanje labela...")
        self.train_df = pd.read_csv(self.labels_file)
        print(f"Ucitano {len(self.train_df)} labela")
        return self.train_df
    
    def validate_data(self) -> Dict:
        print("\nValidacija podataka...")

        results = {
            'total_train_labels': len(self.train_df),
            'unique_labels': self.train_df['label'].nunique(),
            'label_distribution': self.train_df['label'].value_counts().to_dict(),
            'missing_labels': self.train_df['label'].isna().sum(),
            'duplicate_ids': self.train_df['id'].duplicated().sum(),
        }

        #Provera da li postoje slike za sve ID-jeve
        print("Provera postojanja slika...")
        missing_images = []
        sample_size = min(1000, len(self.train_df))

        for idx in self.train_df['id'].head(sample_size):
            img_path = self.train_dir / f"{idx}.png"
            if not img_path.exists():
                missing_images.append(idx)

        results['missing_images_sample'] = len(missing_images)
        results['sample_checked'] = sample_size

        #Provera dimenzija slika
        print("Provera dimenzija slika...")
        image_dims = []
        for idx in self.train_df['id'].head(100):
            img_path = self.train_dir / f"{idx}.png"
            if img_path.exists():
                img = Image.open(img_path)
                image_dims.append(img.size)

        results['image_dimensions'] = {
            'unique_dims': list(set(image_dims)),
            'most_common': max(set(image_dims), key=image_dims.count) if image_dims else None
        }

        self.validation_results = results
        
        print(f"Validacija završena")
        print(f"  - Ukupno labela: {results['total_train_labels']}")
        print(f"  - Jedinstvenih klasa: {results['unique_labels']}")
        print(f"  - Nedostajućih labela: {results['missing_labels']}")
        print(f"  - Duplikata ID-jeva: {results['duplicate_ids']}")
        
        return results
    
    def clean_data(self) -> pd.DataFrame:
        print("\nČišćenje podataka...")

        initial_count = len(self.train_df)
        
        # Uklanjanje duplikata
        self.train_df = self.train_df.drop_duplicates(subset=['id'])
        
        # Uklanjanje redova sa nedostajućim labelama
        self.train_df = self.train_df.dropna(subset=['label'])
        
        # Provera da li su sve labele validne
        valid_labels = self.train_df['label'].isin(self.classes)
        invalid_count = (~valid_labels).sum()
        
        if invalid_count > 0:
            print(f"  ⚠ Pronađeno {invalid_count} nevalidnih labela")
            self.train_df = self.train_df[valid_labels]
        
        final_count = len(self.train_df)
        removed = initial_count - final_count
        
        print(f"Čišćenje završeno")
        print(f"  - Uklonjeno redova: {removed}")
        print(f"  - Preostalo redova: {final_count}")
        
        return self.train_df
    
    def get_descriptive_statistics(self) -> Dict:
        print("\nRačunanje deskriptivne statistike...")

        stats = {
            'total_samples': len(self.train_df),
            'num_classes': self.train_df['label'].nunique(),
            'class_distribution': self.train_df['label'].value_counts().to_dict(),
            'class_percentages': (self.train_df['label'].value_counts(normalize=True) * 100).to_dict(),
            'min_class_count': self.train_df['label'].value_counts().min(),
            'max_class_count': self.train_df['label'].value_counts().max(),
            'mean_class_count': self.train_df['label'].value_counts().mean(),
            'std_class_count': self.train_df['label'].value_counts().std(),
        }
        
        print(f"Statistika izračunata")
        print(f"  - Ukupno uzoraka: {stats['total_samples']}")
        print(f"  - Broj klasa: {stats['num_classes']}")
        print(f"  - Prosečan broj uzoraka po klasi: {stats['mean_class_count']:.2f}")
        
        return stats

    def load_sample_images(self, n_samples: int = 10) -> List[Tuple[np.ndarray, str]]:
        print(f"\nUčitavanje {n_samples} slika po klasi...")
        
        samples = []
        
        for class_name in self.classes:
            class_samples = self.train_df[self.train_df['label'] == class_name].head(n_samples)
            
            for _, row in class_samples.iterrows():
                img_path = self.train_dir / f"{row['id']}.png"
                if img_path.exists():
                    img = Image.open(img_path)
                    img_array = np.array(img)
                    samples.append((img_array, row['label']))
        
        print(f"Učitano {len(samples)} slika")
        return samples

    def analyze_image_properties(self, n_samples: int = 500) -> Dict:
        print(f"\nAnaliza svojstava slika (uzorak od {n_samples})...")
        
        dimensions = []
        channels = []
        pixel_means = []
        pixel_stds = []
        
        sample_ids = self.train_df['id'].sample(min(n_samples, len(self.train_df)))
        
        for img_id in sample_ids:
            img_path = self.train_dir / f"{img_id}.png"
            if img_path.exists():
                img = Image.open(img_path)
                img_array = np.array(img)
                
                dimensions.append(img_array.shape[:2])
                channels.append(img_array.shape[2] if len(img_array.shape) == 3 else 1)
                pixel_means.append(img_array.mean())
                pixel_stds.append(img_array.std())
        
        analysis = {
            'unique_dimensions': list(set(dimensions)),
            'most_common_dimension': max(set(dimensions), key=dimensions.count),
            'channel_counts': {ch: channels.count(ch) for ch in set(channels)},
            'pixel_value_stats': {
                'mean': np.mean(pixel_means),
                'std': np.mean(pixel_stds),
                'min_mean': np.min(pixel_means),
                'max_mean': np.max(pixel_means),
            },
            'samples_analyzed': len(dimensions)
        }
        
        print(f"Analiza završena")
        print(f"  - Najčešća dimenzija: {analysis['most_common_dimension']}")
        print(f"  - Prosečna vrednost piksela: {analysis['pixel_value_stats']['mean']:.2f}")
        
        return analysis

    def save_processed_data(self, output_path: str = 'processed_train_labels.csv'):
        print(f"\nČuvanje obrađenih podataka u {output_path}...")
        self.train_df.to_csv(output_path, index=False)
        print(f"Podaci sačuvani")

    def run_full_pipeline(self) -> Dict:
        print("=" * 60)
        print("Pokretanje kompletnog Data Pipeline-a")
        print("=" * 60)
        
        # 1. Učitavanje
        self.load_labels()
        
        # 2. Validacija
        validation_results = self.validate_data()
        
        # 3. Čišćenje
        self.clean_data()
        
        # 4. Deskriptivna statistika
        stats = self.get_descriptive_statistics()
        
        # 5. Analiza slika
        image_analysis = self.analyze_image_properties()
        
        # 6. Čuvanje
        self.save_processed_data()
        
        print("\n" + "=" * 60)
        print("Pipeline uspešno završen!")
        print("=" * 60)
        
        return {
            'validation': validation_results,
            'statistics': stats,
            'image_analysis': image_analysis
        }

if __name__ == "__main__":
    # Primer korišćenja
    pipeline = CIFAR10DataPipeline()
    results = pipeline.run_full_pipeline()

