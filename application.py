import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib 
from flask import Flask, request, render_template

# Initialize Flask application
app = Flask(__name__)

# Load the pre-trained model and scaler
model = joblib.load(os.path.join('artifacts', 'model.pkl'))
scaler = joblib.load(os.path.join('artifacts', 'preprocessor.pkl'))

# Define the home page route
@app.route('/')
def home_page():
    return render_template('index.html')

# Define the predict route
@app.route('/predict', methods=['GET', 'POST'])
def predict_datapoint():
    # ...
    if request.method == 'GET':
        return render_template('form.html')
    else:
        try:
            # Collect user input from the form
            carat = float(request.form.get('carat'))
            depth = float(request.form.get('depth'))
            table = float(request.form.get('table'))
            x = float(request.form.get('x'))
            y = float(request.form.get('y'))
            z = float(request.form.get('z'))
            cut = request.form.get('cut')
            color = request.form.get('color')
            clarity = request.form.get('clarity')

            # Create a DataFrame from the user input
            user_data = pd.DataFrame({
                'carat': [carat],
                'depth': [depth],
                'table': [table],
                'x': [x],
                'y': [y],
                'z': [z],
                'cut': [cut],
                'color': [color],
                'clarity': [clarity]
            })

            # Scale the user input using the loaded scaler
            scaled_data = scaler.transform(user_data)

            # Make a prediction using the loaded model
            prediction = model.predict(scaled_data)

            # Round the prediction to two decimal places
            result = round(prediction[0], 2)

            return render_template('form.html', final_result=result)
        except Exception as e:
            # Handle exceptions and log them
            print("Exception occurred:", str(e))
            return "An error occurred while processing the prediction."


if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=True)
