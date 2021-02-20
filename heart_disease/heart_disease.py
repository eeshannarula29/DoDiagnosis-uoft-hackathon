from keras.models import load_model, Model


def build_heart_disease_model() -> Model:
    return load_model('models/heart_disease.h5')


def predict_heart_disease(input: list, model: Model) -> str:
    prediction = model.predict([input])[0]
    if prediction == 0:
        return 'NORMAL'
    else:
        return 'POSITIVE FOR HEART DISEASE'
