from json import load
from flask import Flask, request, jsonify
from sklearn.externals import joblib

app = Flask(__name__)

@app.route("/predict", methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            data = request.get_json()
            years_of_experience = float(data["yearsOfExperience"])
            
            lin_reg = joblib.load("./pickle/linear_regression_model.pkl")
        except ValueError:
            return jsonify("Please enter a number.")

        return jsonify(lin_reg.predict(years_of_experience).tolist())


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000)

