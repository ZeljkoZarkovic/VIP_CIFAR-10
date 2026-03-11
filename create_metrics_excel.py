#Kreiranje Excel fajla sa metrikama svih iteracija modela

import pandas as pd
import json
from pathlib import Path
import glob

print("=" * 70)
print("KREIRANJE EXCEL FAJLA SA METRIKAMA")
print("=" * 70)

#Inicijalizacija
all_metrics = []

#UČITAVANJE REZULTATA CROSS-VALIDACIJE
print("\nUčitavanje rezultata cross-validacije...")
cv_files = glob.glob('logs/cv_results_*.json')

for cv_file in cv_files:
    with open(cv_file, 'r') as f:
        cv_data = json.load(f)
    
    #Dodavanje svakog fold-a kao poseban red
    for fold_result in cv_data['fold_results']:
        row = {
            'Eksperiment': 'Cross-Validation',
            'Model': 'Baseline CNN',
            'Fold': fold_result['fold'],
            'Epoha': fold_result['epochs_trained'],
            'Train Loss': fold_result['train_loss'],
            'Train Accuracy': fold_result['train_accuracy'],
            'Val Loss': fold_result['val_loss'],
            'Val Accuracy': fold_result['val_accuracy'],
            'Vreme Treniranja (s)': fold_result.get('train_time', 'N/A'),
            'Vreme Inferencije (s)': fold_result.get('inference_time', 'N/A'),
        }
        all_metrics.append(row)
    
    #Dodavanje prosečne vrednosti
    row = {
        'Eksperiment': 'Cross-Validation',
        'Model': 'Baseline CNN',
        'Fold': 'PROSEK',
        'Epoha': '-',
        'Train Loss': '-',
        'Train Accuracy': '-',
        'Val Loss': cv_data['avg_val_loss'],
        'Val Accuracy': cv_data['avg_val_accuracy'],
        'Vreme Treniranja (s)': '-',
        'Vreme Inferencije (s)': '-',
    }
    all_metrics.append(row)

print(f"Učitano {len(cv_files)} CV rezultata")

#UČITAVANJE REZULTATA POREĐENJA MODELA
print("\nUčitavanje rezultata poređenja modela...")
comparison_file = 'logs/model_comparison_results.json'

if Path(comparison_file).exists():
    with open(comparison_file, 'r') as f:
        comparison_data = json.load(f)
    
    for model_result in comparison_data:
        model_name = model_result['model_name']
        
        #Dodavanje svakog fold-a
        for fold_result in model_result['fold_results']:
            row = {
                'Eksperiment': 'Model Comparison',
                'Model': model_name,
                'Fold': fold_result['fold'],
                'Epoha': fold_result['epochs_trained'],
                'Train Loss': fold_result['train_loss'],
                'Train Accuracy': fold_result['train_accuracy'],
                'Val Loss': fold_result['val_loss'],
                'Val Accuracy': fold_result['val_accuracy'],
                'Vreme Treniranja (s)': fold_result.get('train_time', 'N/A'),
                'Vreme Inferencije (s)': fold_result.get('inference_time', 'N/A'),
            }
            all_metrics.append(row)
        
        #Dodavanje prosečne vrednosti
        row = {
            'Eksperiment': 'Model Comparison',
            'Model': model_name,
            'Fold': 'PROSEK',
            'Epoha': '-',
            'Train Loss': '-',
            'Train Accuracy': '-',
            'Val Loss': model_result['avg_val_loss'],
            'Val Accuracy': model_result['avg_val_accuracy'],
            'Vreme Treniranja (s)': model_result.get('avg_train_time', 'N/A'),
            'Vreme Inferencije (s)': model_result.get('avg_inference_time', 'N/A'),
        }
        all_metrics.append(row)
    
    print(f"Učitano {len(comparison_data)} modela")
else:
    print("Fajl sa poređenjem modela nije pronađen")

#KREIRANJE EXCEL FAJLA
print("\nKreiranje Excel fajla...")

if all_metrics:
    df = pd.DataFrame(all_metrics)
    
    #Kreiranje Excel writer-a sa više sheet-ova
    excel_file = 'metrike_modela.xlsx'
    
    with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
        #Sheet 1: Sve metrike
        df.to_excel(writer, sheet_name='Sve Metrike', index=False)
        
        #Sheet 2: Samo proseci
        df_avg = df[df['Fold'] == 'PROSEK'].copy()
        df_avg.to_excel(writer, sheet_name='Prosečne Vrednosti', index=False)
        
        #Sheet 3: Po modelima
        if 'Model Comparison' in df['Eksperiment'].values:
            df_comparison = df[df['Eksperiment'] == 'Model Comparison'].copy()
            df_comparison.to_excel(writer, sheet_name='Poređenje Modela', index=False)
        
        #Sheet 4: Rezime
        if df_avg.shape[0] > 0:
            summary_data = []
            for _, row in df_avg.iterrows():
                summary_data.append({
                    'Model': row['Model'],
                    'Val Accuracy': row['Val Accuracy'],
                    'Val Loss': row['Val Loss'],
                    'Avg Train Time (s)': row['Vreme Treniranja (s)'],
                    'Avg Inference Time (s)': row['Vreme Inferencije (s)']
                })
            
            df_summary = pd.DataFrame(summary_data)
            df_summary = df_summary.sort_values('Val Accuracy', ascending=False)
            df_summary.to_excel(writer, sheet_name='Rezime', index=False)
    
    print(f"Excel fajl kreiran: {excel_file}")
    print(f"Ukupno redova: {len(df)}")
    print(f"Sheet-ovi:")
    print(f"     - Sve Metrike")
    print(f"     - Prosečne Vrednosti")
    print(f"     - Poređenje Modela")
    print(f"     - Rezime")
    
    #Prikaz rezimea
    print("\n" + "=" * 70)
    print("REZIME NAJBOLJIH MODELA")
    print("=" * 70)
    
    if df_avg.shape[0] > 0:
        df_display = df_avg[['Model', 'Val Accuracy', 'Val Loss']].copy()
        df_display = df_display.sort_values('Val Accuracy', ascending=False)
        print(df_display.to_string(index=False))
    
else:
    print("Nema podataka za kreiranje Excel fajla")
    print("Pokreni notebook-ove prvo!")

print("\n" + "=" * 70)
print("Gotovo!")
print("=" * 70)
print(f"\nFajl: metrike_modela.xlsx")