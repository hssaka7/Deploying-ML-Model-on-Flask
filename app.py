from flask import Flask, request
import pickle
app = Flask(__name__)


@app.route("/", methods=['GET'])
def home():
    return "Hello Machine Learning"


@app.route("/predict", methods=['GET','POST'])
def predict_input():
    file_destination = "model/classifier_dump.sav"
    model_loaded = pickle.load(open(file_destination, 'rb'))
    if request.method == "POST":
       data = request.json
       x_test = data.get('x_test')
       y_test =  data.get('y_test')
       accuracy = model_loaded.score(x_test, y_test)
       print(accuracy)
    return "This is the page where the user with input the data and predict"


app.run()
