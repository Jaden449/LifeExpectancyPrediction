# 🧬 Life Expectancy Prediction using Random Forest Regression

## 📌 Overview  
This project predicts life expectancy based on various health and lifestyle factors using a **Random Forest Regression Model**. It processes user-inputted health data, applies machine learning for prediction, and provides personalized **health advisory tips** based on individual conditions.  

The app is built using **Flask**, and it features a web interface with multiple pages to guide users through the prediction process.

---

## ✨ Features  
- ✅ **Predicts Life Expectancy** based on health & lifestyle attributes  
- ✅ **Machine Learning Model:** Random Forest Regressor  
- ✅ **Dynamic Health Advisory System** with personalized tips  
- ✅ **Interactive Flask Web App** with user-friendly interface  
- ✅ **Data Preprocessing & Model Training** with real-world health data  
- ✅ **Custom age-group-based prediction rules** for improved accuracy  

---

## 📊 Dataset  

The project utilizes a **pre-processed CSV dataset** (`modified_life_expectancy_dataset.csv`) that includes:  

- 📌 **Health Parameters:** Age, BMR, Blood Pressure, Height, Weight, etc.  
- 📌 **Disease History:** Diabetes, Cancer, HIV, Stroke, Heart Disease, etc.  
- 📌 **Lifestyle Factors:** Smoking, Alcohol Consumption  

---

## ⚙️ Model Implementation  

### 🔹 Data Preprocessing & Feature Engineering  
- Handles **missing values**  
- Encodes **categorical variables**  
- Selects **relevant features** for better model accuracy  

### 🔹 Feature Selection  
The model uses **16 key features**, including:  
- 🏥 **Health Factors**: Age, Blood Pressure, Height, Weight, BMR  
- ⚕️ **Diseases**: Diabetes, Stroke, Kidney Failure, Tuberculosis, HIV, Cancer  
- 🚬 **Lifestyle**: Smoking, Alcohol Consumption  

### 🔹 Model Training & Prediction  
- 🎯 **Random Forest Regression** model is trained on historical health data.  
- 📈 **Performance Metrics:**  
  - **Mean Squared Error (MSE)**  
  - **Root Mean Squared Error (RMSE)**  
  - **R² Score** for model accuracy evaluation  

### 🔹 Health Advisory System  
- 🏥 Personalized **health tips** are generated based on user inputs to provide lifestyle recommendations and disease management advice.  

---

## 🖥️ Web Application Workflow  

### 🌐 App Structure  
The project includes **four main pages** in the Flask web app:  

1️⃣ **age_group.html** → User selects an age group for tailored predictions.  
2️⃣ **index.html** → Inputs health parameters (age, BMI, disease history, etc.).  
3️⃣ **result.html** → Displays **predicted life expectancy** with model accuracy scores.  
4️⃣ **healthtips.html** → Provides **personalized health recommendations** based on user inputs.  

