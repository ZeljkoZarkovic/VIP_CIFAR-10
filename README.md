# CIFAR-10 Image Classification Project

Projekat za klasifikaciju slika iz CIFAR-10 dataset-a, preuzetog sa kaggle, koricenjem konvolucionih neuronskih mreza (CNN).

## Pregled  
Projekat implementira kompletan pipeline za:
1. <b>Analiza podataka</b> - Ucitavanje, ciscenje, validacija, vizuelizacija  
2. <b>Treniranje modela</b> - CNN arhitektura sa cross-validacijom  
3. <b>Poredjenje modela</b> - 5 raylicitih konfiguracija sa detaljnom analizom

### Dataset: CIFAR-10  
- 50.000 slika za trening (32x32 RGB)
- 300.000 slika za testiranje  
- 10 klasa: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck

## Instalacija
Instalacija zavisnosti: <b>pip install -r requirements.txt</b>