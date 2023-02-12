from keras.layers import Conv2D, Dense, Flatten, MaxPooling2D
from keras.models import Sequential


class BaseModel:
    def __init__(self):
        pass

    def build(self):
        pass

    def train(self, train_generator, **kwargs):
        pass

    def predict(self, data_generator):
        pass

    def predict_to_df(self):
        pass


class KerasConvModel(BaseModel):
    def __init__(self):
        super().__init__()
        self.model = None

    def build(self):
        self.model = Sequential()
        self.model.add(Conv2D(32, (3, 3), input_shape=(28, 28, 1), activation="relu"))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Flatten())
        self.model.add(Dense(128, activation="relu"))
        self.model.add(Dense(10, activation="softmax"))
        self.model.compile(
            loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
        )

    def train(self, train_generator, batch_size=128, epochs=10):
        self.model.fit(
            train_generator,
            epochs=epochs,
            batch_size=batch_size,
            verbose=2,
        )

    def predict(self, data_generator):
        return self.model.predict(data_generator)
