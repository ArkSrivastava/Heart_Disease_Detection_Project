from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)

model = joblib.load("best_model.pkl")

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    probability = None

    if request.method == "POST":
        try:
            # Get inputs from form
            age = float(request.form["age"])
            sex = float(request.form["sex"])
            cp = float(request.form["cp"])
            trestbps = float(request.form["trestbps"])
            chol = float(request.form["chol"])
            fbs = float(request.form["fbs"])
            restecg = float(request.form["restecg"])
            thalach = float(request.form["thalach"])
            exang = float(request.form["exang"])
            oldpeak = float(request.form["oldpeak"])
            slope = float(request.form["slope"])
            ca = float(request.form["ca"])
            thal = float(request.form["thal"])

            # Model input
            features = np.array([[age, sex, cp, trestbps, chol, fbs, restecg,
                                  thalach, exang, oldpeak, slope, ca, thal]])

            # Predict
            pred = model.predict(features)[0]
            proba = model.predict_proba(features)[0][1]

            probability = round(proba * 100, 2)

            if pred == 1:
                prediction = "⚠️ High Risk of Heart Disease"
            else:
                prediction = "✅ Low Risk (Healthy)"

        except Exception as e:
            prediction = "Error: Please enter correct values!"
            probability = None

    return render_template("index.html", prediction=prediction, probability=probability)

if __name__ == "__main__":
    app.run(debug=True)
