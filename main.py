from flask import Flask, render_template, url_for, redirect, request
from typing import Any
import os
from load_models import load_models
from pnumonia_detection_model.pnumonia import predict_pnumonia
from malaria.malaria import predict_malaria
from heart_disease.heart_disease import predict_heart_disease


app = Flask(__name__)


@app.route('/')
def index() -> Any:
    return render_template('index.html')


@app.route('/submit_pnumonia', methods=['GET'])
def submit_pnumonia() -> Any:
    user = request.args.get('pneumonia')
    return redirect(url_for('pnumonia_result', filename=user))


@app.route('/results_pnumonia/<filename>')
def pnumonia_result(filename: str) -> Any:
    path = 'templates/assets/img/' + filename
    if os.path.exists(path):
        model = load_models()['pnumonia']
        return render_template('pnumonia_result.html', result=predict_pnumonia(path, model))
    else:
        return redirect('/')


@app.route('/submit_malaria', methods=['GET'])
def submit_malaria() -> Any:
    user = request.args.get('malaria')
    return redirect(url_for('malaria_result', filename=user))


@app.route('/results_malaria/<filename>')
def malaria_result(filename: str) -> Any:
    path = 'templates/assets/img/' + filename
    if os.path.exists(path):
        model = load_models()['malaria']
        return render_template('pnumonia_result.html', result=predict_malaria(path, model))
    else:
        return request('/')


@app.route('/submit_heart_disease', methods=['GET'])
def submit_heart_disease() -> Any:
    age = int(request.args.get('age'))
    sex = 1 if request.args.get('sex') == 'male' else 0
    bp = int(request.args.get('bp'))
    ch = int(request.args.get('ch'))
    bs = 1 if request.args.get('bs') == 'yes' else 0
    hr = int(request.args.get('hr'))

    return redirect(url_for('heart_disease_result', info=str([age, sex, bp, bs, hr, ch])))


@app.route('/results_heart_disease/<info>')
def heart_disease_result(info: str) -> Any:
    model = load_models()['heart disease']
    prediction = predict_heart_disease(eval(info), model)
    return render_template('pnumonia_result.html', result=prediction)


if __name__ == '__main__':
    app.run(debug=True)
