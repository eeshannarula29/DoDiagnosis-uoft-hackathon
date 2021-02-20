import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from typing import List
from keras.models import Sequential, Model
from keras.layers import Dense
from keras.optimizers import SGD, Adam
from keras.activations import relu, sigmoid


def load_data() -> List[np.array]:
    db = pd.read_csv('heart.csv')

    xs = db.drop([column for column in db.columns
                  if column not in ['age', 'sex', 'trestbps', 'fbs', 'thalach', 'chol']], axis=1).to_numpy()
    ys = db['target'].to_numpy()

    return train_test_split(xs, ys, train_size=.95)


def train() -> None:
    x_train, x_test, y_train, y_test = load_data()

    model = Sequential()

    model.add(Dense(50, activation=relu))
    model.add(Dense(25, activation=relu))
    model.add(Dense(5, activation=relu))
    model.add(Dense(1, activation=sigmoid))

    model.compile(Adam(), loss='MSE', metrics=['accuracy'])

    model.fit(x_train, y_train, epochs=500)

    score, acc = model.evaluate(x_test, y_test)
    print('Test score:', score)
    print('Test accuracy:', acc)
    model.save('heart_disease.h5')


if __name__ == '__main__':

    from keras.models import load_model

    model = load_model('models/heart_disease.h5')
    print(model.summary())
