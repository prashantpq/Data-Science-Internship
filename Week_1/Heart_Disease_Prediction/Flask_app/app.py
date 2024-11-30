from flask import Flask, render_template, request
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)

model = joblib.load('main.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        age = float(request.form['age'])
        sex = int(request.form['sex'])
        cp = int(request.form['cp'])
        chol = float(request.form['chol'])
        thalch = float(request.form['thalch'])
        exang = int(request.form['exang'])
        oldpeak = float(request.form['oldpeak'])
        slope = int(request.form['slope'])
        ca = int(request.form['ca'])
        thal = int(request.form['thal'])

        input_data = np.array([[age, sex, cp, chol, thalch, exang, oldpeak, slope, ca, thal]])

        prediction = model.predict(input_data)

        result = f"Prediction: {prediction[0]}"
        return render_template('index.html', prediction=result)

    except Exception as e:
        return f"Error: {e}"

if __name__ == "__main__":
    app.run(debug=True)
