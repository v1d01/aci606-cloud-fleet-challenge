"""
Exercise 4 training script for ACI606 Cloud Fleet Challenge.
Modelos utilizados: Regresión Logística con regularización Ridge (L2) y Lasso (L1).
"""
from pathlib import Path
import os
import joblib
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, StratifiedShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

DATA = Path("data/training.csv")
MODEL_DIR = Path("models")
MODEL_PATH = MODEL_DIR / "model.joblib"

FEATURES = [
    "speed",
    "battery",
    "motor_temp",
    "vibration",
    "motor_current",
    "obstacle_distance",
]
TARGET = "status"


def main() -> None:
    if not DATA.exists():
        raise SystemExit(
            "Training dataset not found. Run: python simulator/generate_dataset.py"
        )

    df = pd.read_csv(DATA)

    print("=== 1. EDA: Análisis de Distribución de Clases ===")
    print(df[TARGET].value_counts())
    print("\n" + "=" * 50)

    X = df[FEATURES]
    y = df[TARGET]

    # Separación Train/Test agrupada o estratificada
    if "fleet_id" in df.columns:
        print("Usando GroupShuffleSplit basado en 'fleet_id'...")
        gss = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
        train_idx, test_idx = next(gss.split(X, y, df["fleet_id"]))
    elif "robot_id" in df.columns:
        print("Usando GroupShuffleSplit basado en 'robot_id'...")
        gss = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
        train_idx, test_idx = next(gss.split(X, y, df["robot_id"]))
    else:
        print("Usando StratifiedShuffleSplit...")
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
        train_idx, test_idx = next(sss.split(X, y))

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    print(f"Muestras de Entrenamiento: {len(X_train)} | Muestras de Prueba: {len(X_test)}")
    print("=" * 50)

    # 2. Comparación de Regresión Logística con Ridge (L2) vs Lasso (L1)
    models = {
        "LogisticRegression_Ridge_L2": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(penalty="l2", C=1.0, solver="lbfgs", max_iter=1000, random_state=42))
        ]),
        "LogisticRegression_Lasso_L1": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(penalty="l1", C=1.0, solver="saga", max_iter=1000, random_state=42))
        ])
    }

    best_f1 = -1.0
    best_pipeline = None
    best_model_name = ""

    print("\n=== 2. Entrenamiento y Evaluación de Modelos (Ridge vs Lasso) ===")
    for name, pipeline in models.items():
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        # Métrica principal: F1 Macro
        macro_f1 = f1_score(y_test, y_pred, average="macro")

        print(f"\n---> Modelo: {name}")
        print(f"F1 Macro: {macro_f1:.4f}")
        print("\nMatriz de Confusión:")
        labels = ["NORMAL", "WARNING", "CRITICAL"]
        print(pd.DataFrame(
            confusion_matrix(y_test, y_pred, labels=labels),
            index=[f"Real {l}" for l in labels],
            columns=[f"Pred {l}" for l in labels]
        ))
        print("\nReporte de Clasificación:")
        print(classification_report(y_test, y_pred, zero_division=0))
        print("-" * 50)

        if macro_f1 > best_f1:
            best_f1 = macro_f1
            best_pipeline = pipeline
            best_model_name = name

    # Justificación técnica
    print(f"\n=== 3. Modelo Seleccionado: {best_model_name} (F1 Macro: {best_f1:.4f}) ===")

    # Guardar pipeline en models/model.joblib
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_pipeline, MODEL_PATH)
    print(f"Pipeline guardado exitosamente en: {MODEL_PATH}")


if __name__ == "__main__":
    main()