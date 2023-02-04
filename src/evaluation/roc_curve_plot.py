from sklearn.metrics import roc_curve, RocCurveDisplay
import pandas as pd
import matplotlib.pyplot as plt

def roc_curve_plots(preds, labels):
    """Given two dataframes, one containing the predictions and one containing
    the labels, it prints the ROC curve for the resulting patients.

    Args:
        preds: Pandas dataframe with the following two columns `filepath`,
            `prediction`.
        labels: Pandas dataframe with the following two columns `filepath`,
            `label`.

    Returns:
        None
    """
    # Combine `preds` and `labels` into one DataFrame
    df = labels.merge(preds, on="filepath")

    # Add a column to select predictions by patient ID easily
    df["patient_id"] = df["filepath"].apply(lambda fp: fp.split("/")[0])

    # Separate results by patient id
    patient_ids = list(df["patient_id"].unique())

    for patient_id in patient_ids:
        selection = df[df["patient_id"] == patient_id]

        fpr, tpr, _ = roc_curve(selection["label"], selection["prediction"])
        roc_display = RocCurveDisplay(fpr=fpr, tpr=tpr).plot()

        roc_display.ax_.set_title('Patient: ', patient_id)

    plt.show()

    return None