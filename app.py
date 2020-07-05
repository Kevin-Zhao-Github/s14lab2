import joblib
from flask import Flask, render_template
app = Flask(__name__)


@app.route('/')
def hello_world():
    model1 = joblib.load('lin_regr.pkl')
    pred1 = model1.predict([[4, 2.5, 3005, 15, 17903.0, 1]])[0][0].round(1)
    model2 = joblib.load('decision_tree.pkl')
    pred2 = model2.predict([[4, 2.5, 3005, 15, 17903.0, 1]])[0].round(1)
    return render_template('index.html', preds=(pred1, pred2))
