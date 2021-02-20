from pnumonia_detection_model.pnumonia import build_model_pnumonia
from keras.models import Model
from typing import Dict
from malaria.malaria import build_model_malaria
from heart_disease.heart_disease import build_heart_disease_model


def load_models() -> Dict[str, Model]:
    return {'pnumonia': build_model_pnumonia(),
            'malaria': build_model_malaria(),
            'heart disease': build_heart_disease_model()}

