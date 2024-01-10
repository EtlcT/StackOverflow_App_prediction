from flask import Flask, request, jsonify
from predict.predict import run

app = Flask(__name__)


@app.route('/', methods=['GET'])
def hello():
    return '<p>Hello</p>'


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    pass
    try:
        data = request.args.get('title')
        model = run.TextPredictionModel.from_artefacts('../../train/data/artefacts/2024-01-09-15-35-30/')
        predictions = model.predict(data)
        return jsonify({"predicted tags" : predictions})
    except Exception as e:
        return jsonify({"error": str(e)})


if __name__ == '__main__':
    app.run(debug=True)
