from .models import KerasConvModel, BaseModel
import matplotlib.pyplot as plt # Para los plots
import numpy  as np # Para los plots

def plots_history(H,n_epochs):
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

def train_pipeline(model: BaseModel, optimizer_alg, train_gen, valgen, epochs=30, batch_size=200):

    """Train the model."""

    # load data con generador
        # datagen
        # train_generator = datagen.flow_from_directory(
        # val_generator = datagen.flow_from_directory(

    
    # build model
    model.build()

    # train model H = History of training process
    H = model.train(X_train, y_train, X_test, y_test, epochs=epochs, batch_size=batch_size)

    # plot results
    plots_history(H, epochs)

    # evaluate_model(model, X_test, y_test)

    # save model for future use

    #save_model(model, MODEL_FILE_NAME)

    # save evaluation metrics for future use

    #save_evaluation_metrics(model, X_test, y_test, EVALUATION_FILE_NAME)



if __name__ == '__main__':
    my_model = KerasConvModel()
    train_pipeline(my_model)