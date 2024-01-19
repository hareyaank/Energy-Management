import numpy as np
from flask import Flask, request, render_template
import pickle

app = Flask(__name__)

model = pickle.load(open('classifier (2).pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get input features from the form and convert to float
    float_features = [float(x) for x in request.form.values()]
    
    
    final_features = [np.array(float_features)]
       # Make predictions using the loaded model
    pred = model.predict(final_features)
    
    # Pass the prediction to the result template
    return render_template('decision.html', prediction=pred)

if __name__ == "__main__":
    # Run the Flask application in debug mode
    app.run(debug=True)



