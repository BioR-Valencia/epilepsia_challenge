#import matplotlib.pyplot as plt  # Para los plots
import numpy as np  # Para los plots

from src.data.make_dataset import DataLoader

from .models import BaseModel, KerasConvModel, SklearnAdaBoostModel

from ..evaluation import evaluate_report, confusion_matrix_plot, roc_curve_plot

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
):

    train_generator, test_generator = data_loader.get_generators(1110)

    """Train the model."""

    # build model
    model.build()

    # train model H = History of training process
    model.train(train_generator)

    # plot results
    # plots_history(H, epochs)

    # evaluate_model(model, X_test, y_test)
    predict_dfs = model.predict_to_df(test_generator, treshold = 0.5)

    confusion_matrix_plot(predict_dfs['preds'], predict_dfs['labels'])

    roc_curve_plot(predict_dfs['preds'], predict_dfs['labels'])

    report = evaluate_report(predict_dfs['preds'], predict_dfs['labels'])

    print('Resultado AUC: ', report['patient_aucs'],'\n')
    print('Reporte: ', report['patients_report'],'\n')


    # save model for future use

    # save_model(model, MODEL_FILE_NAME)

    # save evaluation metrics for future use

    # save_evaluation_metrics(model, X_test, y_test, EVALUATION_FILE_NAME)


if __name__ == "__main__":
    my_model = KerasConvModel()
    train_pipeline(my_model)
