from sklearn.metrics import ConfusionMatrixDisplay
import pandas as pd
import matplotlib.pyplot as plt

def confusion_matrix_plots(preds, labels, class_names = ['Inter-ictal','Pre-ictal'], normalize = None):
    """Given two dataframes, one containing the predictions and one containing
    the labels, it prints the confusion matrix for the resulting patients.

    Args:
        preds: Pandas dataframe with the following two columns `filepath`,
            `prediction`.
        labels: Pandas dataframe with the following two columns `filepath`,
            `label`.
        class_names: List of strings containing the names of the classes,
        normalize: Boolean, if True, the confusion matrix will be normalized.

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

        disp = ConfusionMatrixDisplay.from_predictions(
         selection["label"], 
         selection["prediction"],
         display_labels=class_names,
         cmap=plt.cm.Blues,
         normalize=normalize,
        )
        disp.ax_.set_title('Patient: {}'.format(patient_id))

    plt.show()

    return None