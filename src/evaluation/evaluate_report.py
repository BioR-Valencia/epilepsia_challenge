import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score


def evaluate(preds, labels):
    """Given two dataframes, one containing the predictions and one containing
    the labels, it returns a report with:

    - Patient-Wise classification report: precision, recall, f1-score, support for 0s & 1s
            *in string format (can be printed)*
    - Patient-Wise ROC AUC
    - Patient-Wise confusion matrix

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
    patient_aucs = {}  # Patient-wise AUC
    patient_matrices = {}  # Patient-wise confusion matrix
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
        "report": patients_report,  # Classification report
        "patient_aucs": patient_aucs,  # Patient-wise AUC scores
        "matrix": patient_matrices,  # Confusion matrix
        "filepaths": df["filepath"].tolist(),  # Filepaths of the samples
    }

    return dict_final
