import requests
from main_script import load_data
import pandas as pd
import json

# server url
URL = "http://localhost:5000/predict"

# load data
DATA_PATH = 'covtype.data'

class_dir = {0: 'Spruce/Fir', 
             1: 'Lodgepole Pine', 
             2: 'Ponderosa Pine', 
             3: 'Cottonwood/Willow', 
             4: 'Aspen', 
             5: 'Douglas-fir', 
             6: 'Krummholz'}

if __name__ == "__main__":

    # open files
    df = pd.read_csv(DATA_PATH, header=None)
    df.drop(df.columns[-1], axis=1, inplace=True)

    # define request payload
    input_features = df.iloc[500,:].tolist() # define input features
    selected_model = 'neural_network' # define selected model

    payload = json.dumps({
        "model": selected_model,
        "input_features": input_features
    })
    
    headers = {
    'Content-Type': 'application/json'
    }

    # Send POST request to API endpoint
    response = requests.post(URL, data=payload, headers=headers)

    # Parse response and print predicted output
    predicted_output = response.content.decode('utf-8')
    print(f'Predicted cover type by {selected_model}:', class_dir[int(predicted_output)])