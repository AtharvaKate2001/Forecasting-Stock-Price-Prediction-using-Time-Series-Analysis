from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

# Load the trained model
model = joblib.load('forecast_model.pkl')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get input values from the form
        input_date = request.form['date']  # Example: '2024-03-23'

        # Preprocess input data (convert to pandas DataFrame)
        input_data = pd.DataFrame({'Date': [input_date]})
        input_data['Date'] = pd.to_datetime(input_data['Date'])

        # Make prediction
        forecast = model.predict(n_periods=1, exogenous=input_data)

        # Format prediction result
        prediction = f'Predicted Price for {input_date}: {forecast[0]:.2f}'

        return render_template('result.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
