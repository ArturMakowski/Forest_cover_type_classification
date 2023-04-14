import requests
from main_script import load_data
import pandas as pd
import json

# server url
URL = "http://localhost:5000/predict"

# load data
DATA_PATH = 'covtype.data'

if __name__ == "__main__":

    # open files
    df = pd.read_csv(DATA_PATH, header=None)
    df.drop(df.columns[-1], axis=1, inplace=True)

    # define request payload
    input_features = df.iloc[507,:].tolist() # define input features
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
    print(f'Predicted output by {selected_model}:', predicted_output)