from keras.models import Sequential


class BaseModel:
    def __init__(self):
        pass

    def build(self):
        pass

    def train(self):
        pass

    def predict(self):
        pass

    def predict_to_df(self):
        pass


class KerasConvModel(BaseModel):
    def __init__(self):
        super().__init__()
        self.model = None

    def build(self):
        self.model = Sequential()
        self.model.add(Conv2D(32, (3, 3), input_shape=(28, 28, 1), activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Flatten())
        self.model.add(Dense(128, activation='relu'))
        self.model.add(Dense(10, activation='softmax'))
        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    def train(self, x_train, y_train, x_test, y_test, epochs=10, batch_size=200):
        self.model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=epochs, batch_size=batch_size, verbose=2)

    def predict(self, x):
        return self.model.predict(x)
