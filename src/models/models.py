# from keras.layers import Conv2D, Dense, Flatten, MaxPooling2D
# from keras.models import Sequential
import numpy as np
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

class BaseModel:
    def __init__(self):
        pass

    def build(self):
        pass

    def train(self, train_generator, **kwargs):
        pass

    def predict(self, data_generator):
        pass

    def predict_to_df(self, data_generator, treshold):
        pass


class SKLearnBaseModel(BaseModel):
    def predict_to_df(self, data_generator, treshold):
        # Given data samples, it returns two dataframes with the predictions and the
        #    truthground labels of the specific samples.
        # Args:
        #     model: the saved trained model to compuct the predictions.
        #     filepath: a string or list of strings with filepaths to the samples to be predicted.
        #     treshold: a float number between 0 and 1. The treshold to convert the predictions to 0s and 1s.
        # Returns:
        #     a list with 2 dataframes. The dataframes are returned in the following format:
        #         df1: | filepath | prediction |
        #         df2: |  filepath | label |

        # Compute predictions
        y_prob, y_test, filepath = self.predict(data_generator)
        # Convert predictions to 0s and 1s
        predictions = y_prob > treshold
        # Create a dataframe with the predictions
        df_preds = pd.DataFrame({"filepath": filepath, "prediction": predictions})
        # Create a dataframe with the labels
        df_labels = pd.DataFrame({"filepath": filepath, "label": y_test})
        # Return the two dataframes
        dict_dfs = {"preds": df_preds, "labels": df_labels}

        return dict_dfs

    @staticmethod
    def iterate_generator(generator):
        X = pd.DataFrame()
        y = []
        filepaths = []
        for idx, (data, label, filepath) in enumerate(generator):
            # if idx + 1 > 200:
            #     break
            # data = [num for num in data]

            # Skip empty data (it was identified as NaN in the feature extraction)
            if data is None:
                continue

            X = pd.concat([X, data], axis=1)
            # print('df: ', X,'\n')
            y.append(label)
            filepaths.append(filepath)

        return X.to_numpy().T, y, filepaths


# clase de un modelo AdaBoost de Sklearn con decision trees como base
class SklearnAdaBoostModel(SKLearnBaseModel):
    def __init__(self):
        super().__init__()
        self.model = None

    def build(self):
        base_estimator = DecisionTreeClassifier()
        self.model = AdaBoostClassifier(estimator=base_estimator, n_estimators=100)

    def train(self, train_generator, **kwargs):

        X_train, y_train, _ = self.iterate_generator(train_generator)
        print('\nTraining shape: ')
        print(X_train.shape)

        self.model.fit(X_train, y_train)

    def predict(self, data_generator):
        X_test, y_test, filepath = self.iterate_generator(data_generator)

        y_prob = self.model.predict_proba(X_test)[:, 1]
        print('\nVal shape: ')
        print(y_prob.shape)

        return y_prob, y_test, filepath

# Clase de un modelo de regresión logística de Sklearn
class SklearnLogisticRegressionModel(SKLearnBaseModel):
    def __init__(self):
        super().__init__()
        self.model = None

    def build(self):
        self.model = LogisticRegression()

    def train(self, train_generator, **kwargs):
        X_train, y_train, _ = self.iterate_generator(train_generator)
        print('\nTraining shape: ')
        print(X_train.shape)

        self.model.fit(X_train, y_train)

    def predict(self, data_generator):
        X_test, y_test, filepath = self.iterate_generator(data_generator)
        y_prob = self.model.predict_proba(X_test)[:, 1]
        print('\nVal shape: ')
        print(y_prob.shape)

        return y_prob, y_test, filepath

# clase de un modelo convolucional de Keras
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

# class fake model to test the pipeline
class FakeModel(SKLearnBaseModel):
    def __init__(self):
        super().__init__()
        self.model = None

    def build(self):
        pass

    def train(self, train_generator, **kwargs):
        pass

    def predict(self, data_generator):
        # the method will be called by superclass method predict_to_df
        # returnin a fake probability prediction, a fake label and a fake filepath
        fake_prob = np.linspace(0.1, 1, 10)

        fake_test = np.array([0,0,1,1,0,0,1,1,1,1]) # 3 true 0, 4 true 1,  1 0 & 2 1 missclassified

        fake_path = list(map('1110/'.__add__,['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']))
        #fake_path -> '1110/' + ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']

        return fake_prob, fake_test, fake_path