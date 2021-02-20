from keras.models import load_model
from keras.models import Model
import numpy as np
import cv2

CATS = ['Parasitized',
        'Uninfected']

DIM1 = DIM2 = 98
streams = 3

SHAPE = shape = (DIM1, DIM2)
SHAPE_streamed = shape_streamed = input_shape = (DIM1, DIM2, streams)

shape_streamed_one = (1, DIM1, DIM2, streams)


def build_model_malaria() -> Model:
    return load_model('models/malaria.h5')


def OneHotEncode(x: np.array, classes: int) -> list:
    targets = []
    for i in x:
        target = [0] * classes
        target[i] = 1
        targets.append(target)
    return targets


def FromOneHot(OneHotEncodedArray: list) -> np.array:
    normal_array = []
    for target in OneHotEncodedArray:
        normal_array.append(list(target).index(max(list(target))))
    return np.array(normal_array)


def make_read_for_input(path: str) -> np.array:
    img = cv2.resize(cv2.imread(path), shape)
    return np.reshape(img, shape_streamed_one) / 255


def predict_malaria(path: str, model: Model) -> str:
    if path is not None:
        inp = make_read_for_input(path)
        prediction = FromOneHot(model.predict(inp))[0]
        return CATS[prediction]
