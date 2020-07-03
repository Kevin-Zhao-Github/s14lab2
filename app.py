import joblib
from flask import Flask, render_template
app = Flask(__name__)


@app.route('/')
def hello_world():
    model = joblib.load('regr.pkl')
    # Make prediction - features = ['BEDS', 'BATHS', 'SQFT', 'AGE', 'LOTSIZE', 'GARAGE']
    prediction = model.predict([[4, 2.5, 3005, 15, 17903.0, 1]])[0][0].round(1)
    prediction = str(prediction)
    return render_template('index.html', pred=prediction)
