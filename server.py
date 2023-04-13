from flask import Flask, request
import numpy as np
import joblib
from main_script import Heuristic, load_data
import tensorflow as tf

app = Flask(__name__)

# Load models
DATA_PATH = '/home/armak/Python_projects_WSL/Forest_cover_type_classification/covtype.data'

X_train, _, _, y_train, _, _ = load_data(DATA_PATH, val_data=True)
heuristic_model = Heuristic(X_train, y_train)
log_model = joblib.load("linear_model.pkl")
knn_model = joblib.load("knn_model.pkl")
nn_model = tf.keras.models.load_model('/home/armak/Python_projects_WSL/Forest_cover_type_classification/nn_model.h5')


@app.route('/predict', methods=['POST'])
def predict():
    # Get selected model from request
    selected_model = request.json['model']
    
    # Get input features from request
    input_features = request.json['input_features']
    
    # Convert input features to numpy array
    input_array = np.array(input_features).reshape((1, -1))
    
    # Make prediction based on selected model
    if selected_model == 'heuristic':
        prediction = heuristic_model.predict(input_array)
    elif selected_model == 'logistic_regression':
        prediction = log_model.predict(input_array)
    elif selected_model == 'knn':
        prediction = knn_model.predict(input_array)
    elif selected_model == 'neural_network':
        prediction = nn_model.predict(input_array)
    else:
        return "Invalid model selected"
    
    return str(prediction)

if __name__ == '__main__':
    app.run()