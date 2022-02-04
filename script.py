import numpy as np
from flask import Flask, request, render_template
import joblib


app = Flask(__name__)
model = joblib.load(open('model.pkl', 'rb'))
cv = joblib.load(open('cv.pkl', 'rb'))


@app.route('/')
def home():
    return render_template('indexmain.html')


@app.route('/predict', methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''

    if request.method == 'POST':
        text = request.form['Review']
        data = [text]
        vectorizer = cv.transform(data).toarray()
        prediction = model.predict(vectorizer)
    if prediction:
        return render_template('indexmain.html', prediction_text='The review is Extreme Positive')
    else:
        return render_template('indexmain.html', prediction_text='The review is Extreme Negative.')


if __name__ == "__main__":
    app.run(debug=True)