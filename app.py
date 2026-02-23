from flask import Flask, render_template, request
import pickle
import pandas as pd
import numpy as np

app = Flask(__name__)

# Load the trained model and scaler
with open('insurance_model1.pkl', 'rb') as f:
    model = pickle.load(f)

with open('insurance_scaler1.pkl', 'rb') as f:
    scaler = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # 1. Extract data from the HTML form
        age = int(request.form['age'])
        sex = request.form['sex']
        bmi = float(request.form['bmi'])
        children = int(request.form['children'])
        smoker = request.form['smoker']
        region = request.form['region']

        # 2. Preprocessing (Must match your Jupyter Notebook steps exactly)
        
        # Map binary variables
        sex_encoded = 1 if sex == 'male' else 0
        smoker_encoded = 1 if smoker == 'yes' else 0
        
        # Handle One-Hot Encoding for Region (drop_first=True meant 'northeast' is all 0s)
        region_northwest = 0
        region_southeast = 0
        region_southwest = 0

        if region == 'northwest':
            region_northwest = 1
        elif region == 'southeast':
            region_southeast = 1
        elif region == 'southwest':
            region_southwest = 1
        # if region is 'northeast', all remain 0

        # 3. Feature Engineering: Calculate Interaction Term
        smoker_bmi = smoker_encoded * bmi

        # 4. Create DataFrame with columns in the EXACT order the scaler expects
        input_data = pd.DataFrame({
            'age': [age],
            'sex': [sex_encoded],
            'bmi': [bmi],
            'children': [children],
            'smoker': [smoker_encoded],
            'region_northwest': [region_northwest],
            'region_southeast': [region_southeast],
            'region_southwest': [region_southwest],
            'smoker_bmi': [smoker_bmi]
        })

        # 5. Scale the data
        scaled_data = scaler.transform(input_data)

        # 6. Predict
        prediction = model.predict(scaled_data)
        output = round(prediction[0], 2)

        return render_template('index.html', prediction_text=f'Estimated Insurance Cost: ${output}')

if __name__ == "__main__":
    app.run(debug=True)