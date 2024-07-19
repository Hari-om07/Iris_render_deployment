from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np

app = Flask(__name__)
model = joblib.load('Task-1.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract form data
        sepal_length = request.form['sepal_length']
        sepal_width = request.form['sepal_width']
        petal_length = request.form['petal_length']
        petal_width = request.form['petal_width']
        
        # Convert to floats
        sepal_length = float(sepal_length)
        sepal_width = float(sepal_width)
        petal_length = float(petal_length)
        petal_width = float(petal_width)
        
        # Create a numpy array
        features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
        
        # Predict using the model
        prediction = model.predict(features)
        return render_template('index.html', prediction_text='Predicted Class: {}'.format(prediction[0]))
    except ValueError:
        return render_template('index.html', prediction_text='Invalid input. Please enter numeric values.')

if __name__ == "__main__":
    app.run(debug=True)
