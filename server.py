from markupsafe import escape
from markupsafe import Markup
from flask import Flask, request
import numpy as np
import joblib
from main_script import Heuristic, load_data
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.model_selection import train_test_split

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():

    # Load data and models
    DATA_PATH = 'covtype.data'

    # Fit scaler on training data
    df = pd.read_csv(DATA_PATH, header=None)
    X = df.iloc[:,:-1]
    y = df.iloc[:,-1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=0)
    sc = StandardScaler().fit(X_train) 

    heuristic_model = Heuristic(X_train, y_train)
    log_model = joblib.load("log_model.pkl")
    dt_model = joblib.load("dt_model.pkl")
    nn_model = tf.keras.models.load_model("nn_model.h5")

    # Get selected model from request
    selected_model = request.json["model"]
    
    # Get input features from request
    input_features = request.json["input_features"]

    # Scale and convert input features to numpy array
    input_array = np.array(input_features).reshape((1, -1))
    input_array = sc.transform(input_array)

    # Make prediction based on selected model
    if selected_model == 'heuristic':
        prediction = heuristic_model.predict(input_array)
    elif selected_model == 'logistic_regression':
        prediction = log_model.predict(input_array)[0]
    elif selected_model == 'decision_tree':
        prediction = dt_model.predict(input_array)[0]
    elif selected_model == 'neural_network':
        prediction = np.argmax(nn_model.predict(input_array))
    else:
        return "Invalid model selected"
    
    return str(prediction)

if __name__ == '__main__':
    app.run()