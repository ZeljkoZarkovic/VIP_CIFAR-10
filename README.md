# CIFAR-10 Image Classification Project

Projekat za klasifikaciju slika iz CIFAR-10 dataset-a, preuzetog sa kaggle, koricenjem konvolucionih neuronskih mreza (CNN).

## <b>Pregled</b>  
Projekat implementira kompletan pipeline za:
1. <b>Analiza podataka</b> - Ucitavanje, ciscenje, validacija, vizuelizacija  
2. <b>Treniranje modela</b> - CNN arhitektura sa cross-validacijom  
3. <b>Poredjenje modela</b> - 5 raylicitih konfiguracija sa detaljnom analizom

### Dataset: CIFAR-10  
- 50.000 slika za trening (32x32 RGB)
- 300.000 slika za testiranje  
- 10 klasa: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck

## <b>Instalacija</b>
Instalacija zavisnosti: <b>pip install -r requirements.txt</b>

## <b>Pokretanje notebook-ova</b>

### Analiza podataka - 01_data_analysis.ipynb  
<b>jupyter notebook 01_data_analysis.ipynb</b>

## <b>Notebook-ovi</b>

### <b>01_data_analysis.ipynb</b>  
Ovaj notebook služi za kompletnu eksploratornu analizu podataka (EDA) CIFAR-10 skupa podataka.  
Koristi se zajedno sa data_pipeline.py i čuva sve rezultate (grafike i CSV fajlove) u folderu analysis_result.

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