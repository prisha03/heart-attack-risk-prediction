# Heart Attack Risk Predictor
Heart Attack Risk Predictor is an end-to-end machine learning (ML) application that uses clinical, lifestyle, and demographic data to classify whether an individual is at high risk of a heart attack. The project was developed to demonstrate skills across the entire data science lifecycle — including data preprocessing, feature engineering, model training, evaluation, and deployment.
The solution is built using Python and integrates a modular pipeline using Scikit-learn, XGBoost, LightGBM, and Logistic Regression. A custom preprocessing pipeline handles missing values, encodes categorical features, and scales numeric attributes using ColumnTransformer and SimpleImputer. The data suffers from class imbalance, which is addressed using SMOTE (Synthetic Minority Oversampling Technique) from the imbalanced-learn library.
Multiple classification models were trained and evaluated using F1 Score, ROC-AUC, Precision, Recall, and Accuracy. The top-performing model is saved and reused in production using joblib.
To provide transparency into how predictions are made, I integrated SHAP (SHapley Additive exPlanations) to generate global and local explanations. These are used to visualize and interpret the most influential features driving heart attack risk.
The entire solution is deployed in a Streamlit web application, where users can input variables such as age, cholesterol, BMI, heart rate, and physical activity, and receive a real-time risk classification and probability. This simulates a user-facing ML-powered health analytics tool — combining both technical accuracy and user accessibility.
This project reflects hands-on experience with:
* Supervised classification models
* Imbalanced dataset handling
* Model explainability
* End-to-end ML deployment
* Frontend integration using Streamlit

# How to Run
1. Clone the repository  
2. Install dependencies:  
   `pip install -r requirements.txt`  
3. Launch the app:  
   `streamlit run heart_attack_predictor_app.py`

# Sample Prediction
Inputs:
- Age: 55
- Cholesterol: 240
- BMI: 28.5
- Physical Activity: 2 days/week
- Systolic BP: 135

Prediction: **High Risk**

# About Me
I'm Prisha Chawla, a Master’s in Business Analytics student at UC Irvine with a background in data analytics and machine learning. I enjoy building real-world solutions and making models accessible to users through thoughtful design and interactivity.

📧 Email: prishachawla10@gmail.com  
🔗 LinkedIn: https://linkedin.com/in/prisha-chawla  
