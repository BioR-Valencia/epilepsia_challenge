from .models import KerasConvModel, BaseModel
from .data import make_data_set
import pandas as pd

def predict_to_df(model: BaseModel, X_test, y_test, filepath, treshold):
    """Given data samples, it returns two dataframes with the predictions and the 
    truthground labels of the specific samples.

    Args:
        model: the saved trained model to compuct the predictions.
        filepath: a string or list of strings with filepaths to the samples to be predicted.

    Returns:
        a list with 2 dataframes. The dataframes are returned in the following format:
            df1: | filepath | prediction | 
            df2: |  filepath | label |
    """
    # Compute predictions
    preds = model.predict(X_test)
    # Convert predictions to 0s and 1s
    preds = [1 if pred > treshold else 0 for pred in preds]
    # Create a dataframe with the predictions
    df_preds = pd.DataFrame({"filepath": filepath, "prediction": preds})
    # Create a dataframe with the labels
    df_labels = pd.DataFrame({"filepath": filepath, "label": y_test})
    # Return the two dataframes
    dict_dfs = {"preds": df_preds, "labels": df_labels}
    
    return dict_dfs
