from flask import Flask, request, jsonify
import pickle
import pandas as pd
from flask_cors import CORS, cross_origin

# Load the saved model from the pickle file
with open('lr_model.pkl', 'rb') as f:
    lr = pickle.load(f)

app = Flask(__name__)
CORS(app)


@app.route('/predict-average-price', methods=['POST'])
@cross_origin()
def predict_average_price():
    """
    API for predicting the average price of avocados
    """
    # Get feature values from the request body
    feature_values = request.json

    # Create a dictionary from the feature values
    input_dict = {
        "Total Volume": feature_values['Total Volume'],
        "4046": feature_values['4046'],
        "4225": feature_values['4225'],
        "4770": feature_values['4770'],
        "Total Bags": feature_values['Total Bags'],
        "Small Bags": feature_values['Small Bags'],
        "Large Bags": feature_values['Large Bags'],
        "XLarge Bags": feature_values['XLarge Bags'],
    }

    # Convert the dictionary into a DataFrame
    input_df = pd.DataFrame([input_dict])

    # Make the prediction using the trained linear regression model
    prediction = lr.predict(input_df)

    # Return the predicted average price in JSON format
    response = jsonify({'predicted_average_price': round(prediction[0], 2)})
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Methods', 'POST')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
    return response


if __name__ == '__main__':
    app.run()
