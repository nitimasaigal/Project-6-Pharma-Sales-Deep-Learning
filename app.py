from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd
#from predict import make_prediction

app = Flask(__name__)

@app.errorhandler(404)
def page_not_found(error):
    return render_template('404.html'), 404

# Load the trained model
model = joblib.load(r"C:\Users\Niti\NEXT HIKES\PROJECT 6\model.pkl")


# Define a route for prediction

@ app.route('/')
def index():
    return render_template('index.html')
    #return 'flask app'

@app.route('/predict', methods=['POST'])
def predict():
    try:

        # Get input data from the request
        input_data = request.get_json(force=True)
        print("Input data:", input_data)
    
        # Convert input data to a DataFrame
        input_df = pd.DataFrame(input_data)
        print("Input DataFrame:", input_df)
    
        # Make prediction
        prediction = model.predict(input_df)
        # Return the prediction as JSON response
        return jsonify({'prediction': prediction.tolist()})
    
    except Exception as e:
        print("An error occurred:", e)
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)