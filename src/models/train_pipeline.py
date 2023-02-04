from .models import KerasConvModel, BaseModel
from .


def train_pipeline(model: BaseModel, x_train, y_train, x_test, y_test, epochs=10, batch_size=200):

    """Train the model."""

    # load data

    data = load_data()

    X_train = data['X_train']

    y_train = data['y_train']

    X_test = data['X_test']

    y_test = data['y_test']

    # train model

    model.build()
    model.train(X_train, y_train, X_test, y_test, epochs=epochs, batch_size=batch_size)

    # evaluate model

    evaluate_model(model, X_test, y_test)

    # save model for future use

    save_model(model, MODEL_FILE_NAME)

    # save evaluation metrics for future use

    save_evaluation_metrics(model, X_test, y_test, EVALUATION_FILE_NAME)



if __name__ == '__main__':
    my_model = KerasConvModel()
    train_pipeline(my_model)