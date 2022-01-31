# Import libraries
import numpy as np
from flask import Flask, request, jsonify
import pickle
import joblib

app = Flask(__name__)
# Load the model
model = joblib.load(open('RlTitanic.pkl','rb'))
@app.route('/api',methods=['POST'])
def predict():
    # Get the data from the POST request.
    #data = request.get_json(force=True)
    data = request.json
    #return data['data']['clase']
    # Make prediction using model loaded from disk as per the data.
    prediction = model.predict(np.array([data['data']['clase'], data['data']['sexo']]).reshape(1, -1))
    # Take the first value of prediction
    output = prediction[0]
    #return str(output)
    return jsonify(int(output))
if __name__ == '__main__':
    app.run(port=5000, debug=True)