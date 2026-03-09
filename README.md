# CIFAR-10 Image Classification Project

Projekat za klasifikaciju slika iz CIFAR-10 dataset-a, preuzetog sa kaggle, koricenjem konvolucionih neuronskih mreza (CNN).

## <b>Pregled</b>  
Projekat implementira kompletan pipeline za:
1. <b>Analiza podataka</b> - Ucitavanje, ciscenje, validacija, vizuelizacija  
2. <b>Treniranje modela</b> - CNN arhitektura sa cross-validacijom  
3. <b>Poredjenje modela</b> - 5 razlicitih konfiguracija sa detaljnom analizom

### Dataset: CIFAR-10  
- 50.000 slika za trening (32x32 RGB)
- 10.000 slika za testiranje  
- 10 klasa: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck

## <b>Instalacija</b>
Instalacija zavisnosti: <b>pip install -r requirements.txt</b>

## <b>Pokretanje notebook-ova</b>

### Analiza podataka - 01_data_analysis.ipynb  
<b>jupyter notebook 01_data_analysis.ipynb</b>

### Model Training - 02_model_training.ipynb
<b>jupyter notebook 02_model_training.ipynb</b>

### Model Comparison - 03_model_comparison.ipynb
<b>jupyter notebook 03_model_comparison.ipynb</b>

## <b>Notebook-ovi</b>

### <b>01_data_analysis.ipynb</b>  
Ovaj notebook služi za kompletnu eksploratornu analizu podataka (EDA) CIFAR-10 skupa podataka.  
Koristi se zajedno sa <b>data_pipeline.py</b> (reproducibilni koraci za ucitavanje, ciscenje, validaciju i transformaciju podataka) i čuva sve rezultate (grafike i CSV fajlove) u folderu <b>analysis_result</b>.

Sekcije:
- Import biblioteka
- Učitavanje podataka
- Validacija
- Provera nedostajucih podataka
- Ciscenje
- Deskriptivna statistika
- Vizualizacija distribucije klasa
- Provera balansiranosti
- Analiza svojstava slika
- Vizualizacija uzoraka
- Analiza distribucije piksela
- Korelaciona analiza
- Čuvanje obrađenih podataka

Vreme izvrsavanja: ~2–3 minuta

### <b>02_model_training.ipynb</b>
Ovaj notebook sluzi za definisanje arhitekture i treniranje modela.  
Koristi se zajedno sa <b>model_arhitecture.py</b> (Definicija arhitekture neuronske mreze u klasfikaciji CIFAR-10 slika) i sa <b>train_model.py</b>  
(treniranje modela sa cross-validacijom, optimizacijom hiperparametara i logovanjem). Rezultati se cuvaju u folderima <b>training_result, logs, mlruns, models</b>.

Sekcije:
- Import biblioteka
- Pregled arhitekture
- Učitavanje podataka
- Vizualizacija uzoraka
- Definisanje hiperparametara
- Treniranje sa cross-validacijom
- Analiza rezultata

Vreme izvrsavanja: ~10-20 minuta (za 5,000 uzoraka)

### <b>03_model_comparison.ipynb</b>  
Ovaj notebook sluzi za poredjenje 5 razlicitih konfiguracija modela.  
Koristi se zajedno sa <b>model_comparison.py</b> (Modul za treniranje i poredjenje razlicitih modela) i <b>evaluation_metrics.py</b> (Dodatne metrike za evaluaciju modela (ROC, PR krive)). Rezultati se cuvaju u folderu <b>logs</b>

Sekcije:
- Definisanje 5 konfiguracija
- Ucitavanje podataka
- Treniranje svih modela
- Tabela poredjenja
- Vizualizacija poredjenja
- Learning curves
- Confusion matrix
- Cuvanje rezultata

Vreme izvrsavanja: ~30-60 minuta (za 10,000 uzoraka) koriscenjem GPU-a, u mom slucaju 5h koriscenjem CPU-a :(