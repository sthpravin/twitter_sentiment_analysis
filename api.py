import flask
from flask import Flask, render_template, request, Response
from sklearn.externals import joblib
from evaluator import evaluate


app = Flask(__name__)



@app.route("/predict2", methods =['POST'])
def predict():
    print("/predict2 called...")
    text = request.form['text']
    print(text)
    prediction =  evaluate(text)
    return str(prediction)

@app.route("/")
@app.route("/index")
def index():
    return "This is index..."

if __name__ == '__main__':
    model = joblib.load('model.pkl')
    app.run(host='0.0.0.0', port=8080, debug=True)