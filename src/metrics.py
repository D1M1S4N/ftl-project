from sklearn.metrics import accuracy_score, f1_score, recall_score, confusion_matrix
import numpy as np

def calculate_classification_metrics(y_true, y_pred):
    """
    Calcula las métricas de dominio para la clasificación EM vs Control.
    1 = Lesión, 0 = Control.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Accuracy y F1-Score ponderado
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    
    # Sensibilidad (Recall de la clase 1)
    # Importante para no ignorar las lesiones
    sensitivity = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
    
    # Especificidad (TN / (TN + FP))
    # Para ver cómo de bien detecta los controles sanos
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    
    return {
        "accuracy": float(acc),
        "f1_score": float(f1),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity)
    }


def calculate_patient_level_metrics(dataframe, y_true, y_pred=None, y_score=None):
    """
    Agrega predicciones de slices a nivel de paciente y calcula las mismas métricas.

    Si se proporcionan probabilidades/scores de la clase EM en ``y_score``, se
    promedian por paciente y se aplica umbral 0.5. Si no, se usa voto mayoritario
    sobre ``y_pred``.
    """
    if y_pred is None and y_score is None:
        raise ValueError("Se debe proporcionar y_pred o y_score para agregar por paciente.")

    eval_df = dataframe[["patient_id"]].copy().reset_index(drop=True)
    eval_df["label_id"] = np.asarray(y_true, dtype=int)

    label_counts = eval_df.groupby("patient_id")["label_id"].nunique()
    if (label_counts > 1).any():
        bad_patients = label_counts[label_counts > 1].index.tolist()
        raise ValueError(f"Pacientes con etiquetas inconsistentes en evaluación: {bad_patients[:5]}")

    if y_score is not None:
        eval_df["score"] = np.asarray(y_score, dtype=float)
        patient_df = (
            eval_df
            .groupby("patient_id", as_index=False)
            .agg(label_id=("label_id", "first"), score=("score", "mean"))
        )
        patient_df["prediction"] = (patient_df["score"] >= 0.5).astype(int)
    else:
        eval_df["prediction"] = np.asarray(y_pred, dtype=int)
        patient_df = (
            eval_df
            .groupby("patient_id", as_index=False)
            .agg(label_id=("label_id", "first"), prediction=("prediction", "mean"))
        )
        patient_df["prediction"] = (patient_df["prediction"] >= 0.5).astype(int)

    return calculate_classification_metrics(
        patient_df["label_id"].to_numpy(),
        patient_df["prediction"].to_numpy(),
    )
