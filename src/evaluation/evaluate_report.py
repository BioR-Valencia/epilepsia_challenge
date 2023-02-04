from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
import pandas as pd

def evaluate(preds, labels):
    """Given two dataframes, one containing the predictions and one containing
    the labels, it returns a report with:

    - Patient-Wise classification report: precision, recall, f1-score, support for 0s & 1s
    - Patient-Wise ROC AUC
    - Patient-Wise confusion matrix
    the Patient-Wise ROC AUC (average value of the
    separate ROC-AUC scores for each patient).

    Args:
        preds: Pandas dataframe with the following two columns `filepath`,
            `prediction`.
        labels: Pandas dataframe with the following two columns `filepath`,
            `label`.

    Returns:
        score: Float
    """

    # Combine `preds` and `labels` into one DataFrame
    df = labels.merge(preds, on="filepath")

    # Add a column to select predictions by patient ID easily
    df["patient_id"] = df["filepath"].apply(lambda fp: fp.split("/")[0])

    # Separate results by patient id
    patient_ids = list(df["patient_id"].unique())

    patients_report = {}  # Patient-wise classification report
    patient_aucs = {} # Patient-wise AUC
    patient_matrices = {} # Patient-wise confusion matrix
    for patient_id in patient_ids:
        selection = df[df["patient_id"] == patient_id]
        # Patient-wise classification report
        patients_report[patient_id] = classification_report(
            selection["label"], selection["prediction"]
        )
        # Patient-wise AUC
        patient_aucs[patient_id] = roc_auc_score(
            selection["label"], selection["prediction"]
        )
        # Patient-wise confusion matrix
        patient_matrices[patient_id] = confusion_matrix(
            selection["label"], selection["prediction"]
        )

    dict_final = {
        "report": report,   # Classification report
        "patient_aucs": patient_aucs,     # Patient-wise AUC scores
        "matrix": matrix,   # Confusion matrix
    }

    return dict_final