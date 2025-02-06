import pandas as pd
import numpy as np
import streamlit as st
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
import xgboost as xgb

# Load data
df = pd.read_csv('datasauxuly.csv')
X = df[['gender', 'age', 'hypertension', 'heart_disease', 'ever_married', 'work_type', 'Residence_type', 'avg_glucose_level', 'bmi', 'smoking_status']]
y = df['stroke']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=True, random_state=42)

# Train models
decision_tree = DecisionTreeClassifier(criterion='gini', max_depth=10, min_samples_split=2, min_samples_leaf=1, random_state=42)
decision_tree.fit(X_train, y_train)

neural_network_model = MLPClassifier(max_iter=300, hidden_layer_sizes=(100, 100), alpha=0.0001, learning_rate='constant', solver='adam', activation='relu', random_state=42)
neural_network_model.fit(X_train, y_train)

svc_model = SVC(kernel='rbf', C=1.0, random_state=42)
svc_model.fit(X_train, y_train)

gradient_boosting_model = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=3, random_state=42)
gradient_boosting_model.fit(X_train, y_train)

xgboost_model = xgb.XGBClassifier(max_depth=3, learning_rate=0.1, n_estimators=100, random_state=42)
xgboost_model.fit(X_train, y_train)

# Streamlit interface
st.title("Dự đoán đột quỵ")

st.header("Nhập thông tin:")

gender = st.selectbox("Giới tính", [0, 1])
age = st.number_input("Tuổi", min_value=0, max_value=120, value=25)
hypertension = st.selectbox("Hypertension (0/1)", [0, 1])
heart_disease = st.selectbox("Heart Disease (0/1)", [0, 1])
ever_married = st.selectbox("Đã từng kết hôn (0/1)", [0, 1])
work_type = st.selectbox("Loại công việc", [0, 1, 2, 3, 4])
residence_type = st.selectbox("Loại khu vực sinh sống (0/1)", [0, 1])
avg_glucose_level = st.number_input("Avg Glucose Level", min_value=0.0, max_value=300.0, value=100.0)
bmi = st.number_input("BMI", min_value=0.0, max_value=100.0, value=25.0)
smoking_status = st.selectbox("Smoking Status (0/1)", [0, 1])

input_data = np.array([[gender, age, hypertension, heart_disease, ever_married, work_type, residence_type, avg_glucose_level, bmi, smoking_status]])

def predict():
    predictions = {
        "Decision Tree": decision_tree.predict(input_data)[0],
        "Neural Network": neural_network_model.predict(input_data)[0],
        "SVC": svc_model.predict(input_data)[0],
        "Gradient Boosting": gradient_boosting_model.predict(input_data)[0],
        "XGBoost": xgboost_model.predict(input_data)[0]
    }

    result = "\n".join([f"{model}: {'Đột quỵ' if pred == 1 else 'Không đột quỵ'}" for model, pred in predictions.items()])
    st.success(f"Kết quả dự đoán:\n{result}")

if st.button("Dự đoán"):
    predict()
