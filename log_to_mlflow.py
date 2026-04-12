#LOGOVANJE POSTOJEĆIH REZULTATA U MLFLOW

import mlflow
import json
import glob
from pathlib import Path

print("=" * 70)
print("LOGOVANJE REZULTATA U MLFLOW")
print("=" * 70)

#PATH SETUP
BASE_PATH = r"C:\Users\milan\OneDrive\Радна површина\VIP_CIFAR-10\logs"

cv_files = glob.glob(f"{BASE_PATH}\cv_results_*.json")
comparison_file = f"{BASE_PATH}\model_comparison_results.json"

#MLFLOW SETUP
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("cifar10_experiments")

print("CV files:", cv_files)
print("Comparison exists:", Path(comparison_file).exists())
print("Tracking URI:", mlflow.get_tracking_uri())

#CV REZULTATI
for cv_file in cv_files:

    with open(cv_file, "r") as f:
        cv_data = json.load(f)

    with mlflow.start_run(run_name="Baseline_CV"):

        #hiperparametri
        mlflow.log_params(cv_data["config"])
        mlflow.log_param("n_folds", len(cv_data["fold_results"]))

        #fold metrike
        for fold in cv_data["fold_results"]:
            mlflow.log_metrics({
                f"fold_{fold['fold']}_val_accuracy": fold["val_accuracy"],
                f"fold_{fold['fold']}_val_loss": fold["val_loss"],
                f"fold_{fold['fold']}_train_accuracy": fold["train_accuracy"],
                f"fold_{fold['fold']}_train_loss": fold["train_loss"],
            })

        #agregirane metrike
        mlflow.log_metrics({
            "avg_val_accuracy": cv_data["avg_val_accuracy"],
            "avg_val_loss": cv_data["avg_val_loss"],
            "std_val_accuracy": cv_data["std_val_accuracy"],
        })

        #artifact (JSON fajl)
        mlflow.log_artifact(cv_file)

        print(f"✓ Logovan CV (acc: {cv_data['avg_val_accuracy']:.4f})")


#MODEL COMPARISON
if Path(comparison_file).exists():

    with open(comparison_file, "r") as f:
        comparison_data = json.load(f)

    mlflow.set_experiment("cifar10_model_comparison")

    for model_result in comparison_data:

        with mlflow.start_run(run_name=model_result["model_name"]):

            #hiperparametri
            mlflow.log_params(model_result["config"])

            #fold metrike
            for fold in model_result["fold_results"]:
                mlflow.log_metrics({
                    f"fold_{fold['fold']}_val_accuracy": fold["val_accuracy"],
                    f"fold_{fold['fold']}_val_loss": fold["val_loss"],
                    f"fold_{fold['fold']}_train_accuracy": fold["train_accuracy"],
                    f"fold_{fold['fold']}_train_loss": fold["train_loss"],
                })

            #agregirane metrike
            mlflow.log_metrics({
                "avg_val_accuracy": model_result["avg_val_accuracy"],
                "avg_val_loss": model_result["avg_val_loss"],
                "std_val_accuracy": model_result["std_val_accuracy"],
                "avg_train_time": model_result.get("avg_train_time", 0),
                "avg_inference_time": model_result.get("avg_inference_time", 0),
                "model_size_params": model_result.get("model_size_params", 0),
                "model_memory_mb": model_result.get("model_memory_mb", 0),
            })

            #artifact
            mlflow.log_artifact(comparison_file)

            print(f"✓ Logovan model: {model_result['model_name']}")


print("\n" + "=" * 70)
print("GOTOVO")
print("=" * 70)