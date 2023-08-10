# import Flask class from the flask module
from flask import Flask, request
import joblib
import numpy as np

# Create Flask object to run
app = Flask(__name__)

# Load the model from the file
dt_model = joblib.load('D:\exprrf.pkl')

@app.route('/')
def home():
    return "Predict LOS Model Deployment!"

@app.route('/predict')
def predict():
    # Get values from browser
    birth_weight = request.args['birth_weight']
    total_charges = request.args['total_charges']
    #RESP = request.args['RESP']
    #COND = request.args['COND']
    #Mortality = request.args['Mortality']
 
    
    print(birth_weight)

    #test_inp = np.array([birth_weight, total_charges,RESP,COND,Mortality]).reshape(1, 5)
    test_inp = np.array([birth_weight, total_charges]).reshape(1, 2)
    class_predicted = int(dt_model.predict(test_inp)[0])
    output = "Predicted LOS Class: " + str(class_predicted)

    return (output)


if __name__ == "__main__":
    # Start Application
    app.run()