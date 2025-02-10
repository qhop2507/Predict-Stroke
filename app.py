from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)

# Load các mô hình đã huấn luyện
svc_model = joblib.load('models/svc_best_model.pkl') 
mlp_model = joblib.load('models/mlp_model.pkl') 
dt_model = joblib.load('models/dt_best_model.pkl')
gb_model = joblib.load('models/gb_best_model.pkl')
xgb_model = joblib.load('models/xgb_best_model.pkl')

@app.route('/', methods=['GET', 'POST'])
def home():
    prediction_results = None  # Kết quả dự đoán
    if request.method == 'POST':
        try:
            # Nhận dữ liệu từ form
            user_data = np.array([[
                int(request.form['gender']),
                float(request.form['age']),
                int(request.form['hypertension']),
                int(request.form['heart_disease']),
                int(request.form['ever_married']),
                int(request.form['work_type']),
                int(request.form['residence_type']),
                float(request.form['avg_glucose_level']),
                float(request.form['bmi']),
                int(request.form['smoking_status'])
            ]])

            # Dự đoán với tất cả các mô hình
            predictions = {
                "SVC": svc_model.predict(user_data)[0],
                "Neural Network": mlp_model.predict(user_data)[0],
                "Decision Tree": dt_model.predict(user_data)[0],
                "Gradient Boosting": gb_model.predict(user_data)[0],
                "XGBoost": xgb_model.predict(user_data)[0]
            }

            # Chuyển đổi kết quả 0/1 thành chữ
            prediction_results = {model: ("Đột quỵ" if pred == 1 else "Không đột quỵ") for model, pred in predictions.items()}
        
        except ValueError:
            prediction_results = {"Lỗi": "Vui lòng nhập đầy đủ thông tin hợp lệ!"}

    return render_template('index.html', prediction_results=prediction_results)

if __name__ == '__main__':
    app.run(debug=True)
