from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

# Load ML model
with open("model/crop_model.pkl", "rb") as file:
    model = pickle.load(file)

@app.route("/", methods=["GET", "POST"])
def home():
    result = None

    if request.method == "POST":
        N = float(request.form["N"])
        P = float(request.form["P"])
        K = float(request.form["K"])
        temperature = float(request.form["temperature"])
        humidity = float(request.form["humidity"])
        ph = float(request.form["ph"])
        rainfall = float(request.form["rainfall"])

        prediction = model.predict([[N, P, K, temperature, humidity, ph, rainfall]])
        result = prediction[0]

    return render_template("index.html", result=result)

if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=5000)
