import joblib
from flask import Flask, request, render_template
app = Flask(__name__)


@app.route('/', methods=['GET'])
def hello_world():
    return render_template('index.html')


@app.route('/', methods=['POST'])
def predict():
    features = ['BEDS', 'BATHS', 'SQFT', 'AGE', 'LOTSIZE', 'GARAGE']
    feature_values = []
    stats = joblib.load('stats.pkl')
    for i, f in enumerate(features):
        feature_value = float(request.form[f])
        feature_value = (feature_value - stats[f'{f}_mean']) / stats[f'{f}_std']
        feature_value = (feature_value - stats[f'{f}_min']) / (stats[f'{f}_max'] - stats[f'{f}_min'])
        feature_values.append(feature_value)

    pred = 0
    if 'b1' in request.form:
        model = joblib.load('single_tree.pkl')
        pred = model.predict([feature_values])[0].round(1)
    elif 'b2' in request.form:
        model = joblib.load('ensembled_tree.pkl')
        pred = model.predict([feature_values])[0].round(1)
    else:
        model = joblib.load('lin_regr.pkl')
        pred = model.predict([feature_values])[0][0].round(1)

    pred = pred * (stats['SOLDPRICE_max'] - stats['SOLDPRICE_min']) + stats['SOLDPRICE_min']
    pred = pred * stats['SOLDPRICE_std'] + stats['SOLDPRICE_mean']
    return render_template('index.html', preds=pred)
