# import matplotlib.pyplot as plt  # Para los plots
import numpy as np  # Para los plots

from src.data.make_dataset import DataLoader
from src.features.build_features import FeatExtractor
from src.features.Preprocessor import Preprocessor

from ..evaluation import confusion_matrix_plot, evaluate_report, roc_curve_plot
from .models import BaseModel, KerasConvModel, SklearnAdaBoostModel


def plots_history(H, n_epochs):
    # Gráficas
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(np.arange(0, n_epochs), H.history["loss"], label="train_loss")
    plt.plot(np.arange(0, n_epochs), H.history["val_loss"], label="val_loss")
    plt.plot(np.arange(0, n_epochs), H.history["accuracy"], label="train_acc")
    plt.plot(np.arange(0, n_epochs), H.history["val_accuracy"], label="val_acc")
    plt.title("Training Loss and Accuracy")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend()
    plt.show()


def train_pipeline(
    data_loader: DataLoader,
    model: BaseModel,
    patient_id: int = 1110,
    visualize: bool = False,
    labels_prop: float = 2
):

    feature_extraction_functions = [
        (Preprocessor().remove_nans, {}),
        (Preprocessor().convert_time_to_radians, {}),
        (FeatExtractor().mean, {}),
        (FeatExtractor().std, {}),
    ]

    print('Calling data from patient: ' + str(patient_id) + '... \n')

    train_generator, test_generator = data_loader.get_generators(
        patient_id, labels_prop, feature_extraction_functions=feature_extraction_functions
    )

    """Train the model."""

    # build model
    model.build()

    # train model H = History of training process
    model.train(train_generator)

    # plot results
    # plots_history(H, epochs)

    # evaluate_model(model, X_test, y_test)
    predict_dfs = model.predict_to_df(test_generator, treshold=0.5)

    if visualize:
        confusion_matrix_plot.confusion_matrix_plots(
            predict_dfs["preds"], predict_dfs["labels"]
        )

        roc_curve_plot.roc_curve_plots(predict_dfs["preds"], predict_dfs["labels"])

    report = evaluate_report.evaluate(predict_dfs["preds"], predict_dfs["labels"])

    print("Resultado AUC: ", report["patient_aucs"], "\n")

    # save model for future use

    # save_model(model, MODEL_FILE_NAME)

    # save evaluation metrics for future use

    # save_evaluation_metrics(model, X_test, y_test, EVALUATION_FILE_NAME)
    return model, report


if __name__ == "__main__":
    my_model = KerasConvModel()
    train_pipeline(my_model)
