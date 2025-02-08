from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

app = Flask(__name__)
formData={}

# Load the dataset
data = pd.read_csv('modified_life_expectancy_dataset.csv')

# Features and target (Updated to include new features)
X = data[['age', 'bmr', 'diabetes', 'blood_pressure', 'height', 'weight', 'smoking', 'alcohol',
          'hiv', 'cancer', 'hepatitis_b', 'meningitis', 'kidney_failure', 'stroke', 'heartdisease', 'tuberculosis']]
y = data['life_expectancy']

# Split the data for training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Random Forest model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Calculate model accuracy metrics
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)


@app.route('/')
def age_group():
    return render_template('age_group.html')  # First page to select age group


@app.route('/process_age_group', methods=['POST'])
def process_age_group():
    age_group = request.form['age_group']
    if age_group in ['1-20', '20-60', '60-90']:
        return redirect(url_for('index'))  # Redirect to index.html for valid age groups
    else:
        message = "This app is not recommended for individuals aged 90 and above."
        return render_template('age_group.html', message=message)


@app.route('/index')
def index():
    return render_template('index.html')  # Form to input user data


@app.route('/predict', methods=['POST'])
def predict():
    # Collect input data from the form
    age = float(request.form['age'])
    bmr = float(request.form['bmr'])
    diabetes = int(request.form['diabetes'])
    blood_pressure = float(request.form['blood_pressure'])
    height = float(request.form['height'])
    weight = float(request.form['weight'])
    smoking = int(request.form['smoking'])
    alcohol = int(request.form['alcohol'])
    hiv = int(request.form['hiv'])
    cancer = int(request.form['cancer'])
    hepatitis_b = int(request.form['hepatitis_b'])
    meningitis = int(request.form['meningitis'])
    kidney_failure = int(request.form['kidney_failure'])
    stroke = int(request.form['stroke'])
    heartdisease = int(request.form['heartdisease'])
    tuberculosis = int(request.form['tuberculosis'])

    formData["age"]=age
    formData["bmr"]=bmr
    formData["diabetes"]=diabetes
    formData["blood_pressure"]=blood_pressure
    formData["height"]=height
    formData["weight"]=weight
    formData["smoking"]=smoking
    formData["alcohol"]=alcohol
    formData["hiv"]=hiv
    formData["cancer"]=cancer
    formData["hepatitis_b"]=hepatitis_b
    formData["meningitis"]=meningitis
    formData["kidney_failure"]=kidney_failure
    formData["stroke"]=stroke
    formData["heartdisease"]=heartdisease
    formData["tuberculosis"]=tuberculosis
    

    # Prepare input data for prediction
    input_data = [[age, bmr, diabetes, blood_pressure, height, weight, smoking, alcohol, hiv, cancer,
                   hepatitis_b, meningitis, kidney_failure, stroke, heartdisease, tuberculosis]]
    

    # Conditional logic for specific ages
    if 60 <= age < 70:
        base_life_expectancy = 70
        diseases = diabetes + kidney_failure + (cancer if cancer > 0 else 0) + hiv + tuberculosis + smoking + alcohol
        prediction = max(60, base_life_expectancy - (diseases * 0.20))
    elif 70 <= age < 80:
        base_life_expectancy = 80
        diseases = diabetes + kidney_failure + (cancer if cancer > 0 else 0) + hiv + tuberculosis + smoking + alcohol
        prediction = max(70, base_life_expectancy - (diseases * 0.20))
    elif 80 <= age < 90:
        base_life_expectancy = 90
        diseases = diabetes + kidney_failure + (cancer if cancer > 0 else 0) + hiv + tuberculosis + smoking + alcohol
        prediction = max(80, base_life_expectancy - (diseases * 0.20))
    elif age == 90:
        base_life_expectancy = 92
        diseases = diabetes + kidney_failure + (cancer if cancer > 0 else 0) + hiv + tuberculosis + smoking + alcohol
        prediction = max(90, base_life_expectancy - (diseases * 0.20))
    else:
        prediction = model.predict(input_data)[0]

    # Generate health tips based on age and diseases
    health_tips = generate_health_tips(age, diabetes, smoking, alcohol, blood_pressure, height, weight, hiv,
                                       cancer, hepatitis_b, meningitis, kidney_failure, stroke, heartdisease, tuberculosis)

    # Return the result page with prediction and health tips
    return render_template('result.html', prediction=prediction, mse=mse, rmse=rmse, r2=r2, health_tips=health_tips)



def generate_health_tips(age, diabetes, smoking, alcohol, blood_pressure, height, weight, hiv, cancer, 
                         hepatitis_b, meningitis, kidney_failure, stroke, heartdisease, tuberculosis):
    tips = []
    
    # Age-based tips
    if age < 20:
        tips.append("*Age Group Tips:*")
        tips.append("➡️ Focus on balanced nutrition for healthy growth.")
        tips.append("➡️ Engage in physical activities to enhance strength and flexibility.")
        tips.append("➡️ Regular health check-ups are essential during growth years.")
    elif 20 <= age < 60:
        tips.append("*Age Group Tips:*")
        tips.append("➡️ Maintain a healthy work-life balance and manage stress effectively.")
        tips.append("➡️ Adopt a balanced diet with proper nutrients for sustained energy.")
        tips.append("➡️ Routine health check-ups are crucial for early detection of health issues.")
    else:
        tips.append("*Age Group Tips:*")
        tips.append("➡️ Focus on mobility exercises to maintain independence.")
        tips.append("➡️ A diet low in sodium and saturated fats is recommended.")
        tips.append("➡️ Stay socially active to maintain mental well-being.")

    # Disease-specific tips
    if diabetes:
        tips.append("*Diabetes Tips:*")
        tips.append("➡️ Monitor blood sugar levels regularly to manage diabetes effectively.")
        tips.append("➡️ Include foods with a low glycemic index in your diet.")
        tips.append("➡️ Engage in regular physical activities to improve insulin sensitivity.")

    if smoking:
        tips.append("*Smoking Tips:*")
        tips.append("➡️ Quitting smoking reduces the risk of lung and cardiovascular diseases.")
        tips.append("➡️ Seek support groups or counseling for help with smoking cessation.")
        tips.append("➡️ Replace smoking with healthier habits like deep breathing exercises.")

    if alcohol:
        tips.append("*Alcohol Tips:*")
        tips.append("➡️ Limit alcohol intake to recommended levels to protect your liver.")
        tips.append("➡️ Opt for non-alcoholic beverages as alternatives in social settings.")
        tips.append("➡️ Stay hydrated with water to reduce cravings for alcohol.")

    if blood_pressure > 130:
        tips.append("*Blood Pressure Tips:*")
        tips.append("➡️ Reduce salt intake to help manage blood pressure.")
        tips.append("➡️ Regular cardiovascular exercise can help lower blood pressure.")
        tips.append("➡️ Avoid stress triggers and practice relaxation techniques.")

    if height or weight:
        tips.append("*Height/Weight Tips:*")
        tips.append("➡️ Maintain a BMI within the healthy range for your height.")
        tips.append("➡️ Follow a balanced diet and regular exercise routine to manage weight.")
        tips.append("➡️ Consult a nutritionist for a tailored plan if needed.")

    if hiv:
        tips.append("*HIV Tips:*")
        tips.append("➡️ Adhere strictly to your antiretroviral therapy (ART) regimen.")
        tips.append("➡️ Maintain a healthy lifestyle to strengthen your immune system.")
        tips.append("➡️ Avoid infections by practicing good hygiene and safe interactions.")

    if cancer:
        tips.append("*Cancer Tips:*")
        tips.append("➡️ Follow your oncologist's advice for treatment and lifestyle changes.")
        tips.append("➡️ Eat a nutrient-rich diet to support your body's recovery.")
        tips.append("➡️ Manage stress through relaxation techniques or therapy.")

    if hepatitis_b:
        tips.append("*Hepatitis B Tips:*")
        tips.append("➡️ Regular follow-ups with a healthcare provider are crucial.")
        tips.append("➡️ Avoid alcohol to reduce stress on your liver.")
        tips.append("➡️ Eat a liver-friendly diet rich in antioxidants.")

    if meningitis:
        tips.append("*Meningitis Tips:*")
        tips.append("➡️ Stay hydrated and rest to aid recovery.")
        tips.append("➡️ Take prescribed medications as directed by your doctor.")
        tips.append("➡️ Monitor for symptoms like fever or headache and report them promptly.")

    if kidney_failure:
        tips.append("*Kidney Failure Tips:*")
        tips.append("➡️ Follow a kidney-friendly diet low in sodium, potassium, and phosphorus.")
        tips.append("➡️ Stay hydrated but avoid excessive fluid intake.")
        tips.append("➡️ Regularly attend dialysis sessions if prescribed.")

    if stroke:
        tips.append("*Stroke Tips:*")
        tips.append("➡️ Engage in rehabilitation exercises to regain lost functions.")
        tips.append("➡️ Manage risk factors like hypertension and high cholesterol.")
        tips.append("➡️ Avoid smoking and excessive alcohol consumption.")

    if heartdisease:
        tips.append("*Heart Disease Tips:*")
        tips.append("➡️ Follow a heart-healthy diet rich in fruits, vegetables, and lean protein.")
        tips.append("➡️ Take prescribed medications and attend regular cardiac check-ups.")
        tips.append("➡️ Engage in moderate physical activity as advised by your doctor.")

    if tuberculosis:
        tips.append("*Tuberculosis Tips:*")
        tips.append("➡️ Complete the full course of tuberculosis treatment.")
        tips.append("➡️ Eat a diet rich in protein and nutrients to aid recovery.")
        tips.append("➡️ Practice good respiratory hygiene to prevent spreading the infection.")

    return tips

@app.route('/health_advisory')
def health_advisory():
    # Example of tips for a senior user (replace with dynamic user inputs if needed)
    health_tips = generate_health_tips(formData["age"],formData["alcohol"],formData["smoking"],formData["alcohol"],formData["blood_pressure"],formData["height"],formData["weight"],formData["hiv"],formData["cancer"],formData["hepatitis_b"],formData["meningitis"],formData["kidney_failure"],formData["stroke"],formData["heartdisease"],formData["tuberculosis"])
    return render_template('healthAdvisory.html', health_tips=health_tips)


if __name__ == '__main__':
    app.run(debug=True)