import requests
import json

# Define the input data (features) as a Python dictionary
input_data = [[555,1,8,1,4,2015,1,31,1]]

#def make_prediction(input_data):
    #return "This is your prediction"

# Convert the input data dictionary to JSON
input_json = json.dumps(input_data)

# Make a POST request to the Flask application's /predict endpoint
#url = 'http://serene-caverns-82714.herokuapp.com/predict'
url = 'http://127.0.0.1:5000/predict'
try:
    response =requests.post(url, json=input_data)
    response.raise_for_status()  # Raise an exception for HTTP errors (4xx or 5xx)
    prediction = response.json()
    print("Prediction:", prediction)
except requests.exceptions.HTTPError as e:
    print("HTTP error occurred:", e)
except requests.exceptions.ConnectionError as e:
    print("Connection error occurred. Is the Flask server running?")
except requests.exceptions.Timeout as e:
    print("Timeout error occurred. Check your network connection.")
except requests.exceptions.RequestException as e:
    print("An unexpected error occurred:", e)
except json.decoder.JSONDecodeError as e:
    # Handle JSON decoding error
    print("Error decoding JSON:", e)
    print("Response content:", response.content)

