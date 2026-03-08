#Dodatne metrike za evaluaciju modela (ROC, PR krive)

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.preprocessing import label_binarize
from itertools import cycle

def plot_roc_curves(y_true, y_pred_proba, classes, model_name='Model'):
    #Plot ROC krive za multi-class klasifikaciju

    n_classes = len(classes)
    #Binarizacija labela
    y_true_bin = label_binarize(y_true, classes=range(n_classes))

    #Racunanje ROC krive i AUC za svaku klasu
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_pred_proba[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    #Micro-average ROC curve
    fpr["micro"], tpr["micro"], _ = roc_curve(y_true_bin.ravel(), y_pred_proba.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    #Plot
    plt.figure(figsize=(12, 8))

    #Plot micro-average
    plt.plot(fpr["micro"], tpr["micro"],
             label=f'Micro-average (AUC = {roc_auc["micro"]:.2f})',
             color='deeppink', linestyle=':', linewidth=4)
    
    #Plot za svaku klasu
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'green', 'red',
                   'purple', 'brown', 'pink', 'gray', 'olive'])
    
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                label=f'{classes[i]} (AUC = {roc_auc[i]:.2f})')
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title(f'ROC Curves - {model_name}', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right", fontsize=9)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    return roc_auc

def plot_pr_curves(y_true, y_pred_proba, classes, model_name='Model'):
    #Plot Precision-Recall krive za multi-class klasifikaciju

    n_classes = len(classes)

    #Binarizacija labela
    y_true_bin = label_binarize(y_true, classes=range(n_classes))
    
    #Računanje PR krive i AP za svaku klasu
    precision = dict()
    recall = dict()
    average_precision = dict()
    
    for i in range(n_classes):
        precision[i], recall[i], _ = precision_recall_curve(y_true_bin[:, i], 
                                                            y_pred_proba[:, i])
        average_precision[i] = average_precision_score(y_true_bin[:, i], 
                                                       y_pred_proba[:, i])
    
    #Micro-average
    precision["micro"], recall["micro"], _ = precision_recall_curve(
        y_true_bin.ravel(), y_pred_proba.ravel())
    average_precision["micro"] = average_precision_score(y_true_bin, y_pred_proba,
                                                         average="micro")
    
    #Plot
    plt.figure(figsize=(12, 8))
    
    #Plot micro-average
    plt.plot(recall["micro"], precision["micro"],
             label=f'Micro-average (AP = {average_precision["micro"]:.2f})',
             color='deeppink', linestyle=':', linewidth=4)
    
    #Plot za svaku klasu
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'green', 'red',
                   'purple', 'brown', 'pink', 'gray', 'olive'])
    
    for i, color in zip(range(n_classes), colors):
        plt.plot(recall[i], precision[i], color=color, lw=2,
                label=f'{classes[i]} (AP = {average_precision[i]:.2f})')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title(f'Precision-Recall Curves - {model_name}', fontsize=14, fontweight='bold')
    plt.legend(loc="lower left", fontsize=9)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    return average_precision

def plot_roc_pr_comparison(comparator_results, classes):
    #Poredjenje ROC i PR krivih za sve modele

    fig, axes = plt.subplots(len(comparator_results), 2, figsize=(16, 5*len(comparator_results)))
    fig.suptitle('ROC i PR Krive - Svi Modeli', fontsize=16, fontweight='bold')
    
    if len(comparator_results) == 1:
        axes = axes.reshape(1, -1)
    
    for idx, result in enumerate(comparator_results):
        model_name = result['model_name']
        
        # Ovde bi trebalo učitati y_true i y_pred_proba za svaki model
        # Za sada samo placeholder
        print(f"Note: ROC/PR krive za {model_name} zahtevaju dodatne podatke")
    
    plt.tight_layout()
    plt.savefig('logs/roc_pr_curves.png', dpi=300, bbox_inches='tight')
    plt.show()