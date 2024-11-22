from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import xgboost as xgb
import pandas as pd
import joblib
import numpy as np

app = Flask(__name__)

# Load the LSTM and XGBoost models
lstm_model = tf.keras.models.load_model('lstm_model.keras')
xgboost_model = joblib.load('xgboost_model.pkl')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Extract input data from the form
    input_data = {
        'Holiday_Flag': [int(request.form['Holiday_Flag'])],
        'Temperature': [float(request.form['Temperature'])],
        'Fuel_Price': [float(request.form['Fuel_Price'])],
        'CPI': [float(request.form['CPI'])],
        'Unemployment': [float(request.form['Unemployment'])],
        'Year': [int(request.form['Year'])],
        'Month': [int(request.form['Month'])],
        'WeekOfYear': [int(request.form['WeekOfYear'])],
        'DayOfWeek': [int(request.form['DayOfWeek'])],
        'IsWeekend': [int(request.form['IsWeekend'])],
        'Weekly_Sales_Lag_1': [float(request.form['Weekly_Sales_Lag_1'])],
        'Weekly_Sales_MA_4': [float(request.form['Weekly_Sales_MA_4'])]
    }

    res = input_data
    print(res)
    
    # Convert input to DataFrame
    new_data_df = pd.DataFrame(input_data)
    
    # XGBoost prediction
    xgb_dmatrix = xgb.DMatrix(new_data_df)
    xgb_prediction = xgboost_model.predict(xgb_dmatrix)[0]
    
    # LSTM prediction
    lstm_input = np.array(new_data_df.values).reshape(1, new_data_df.shape[0], new_data_df.shape[1])
    lstm_prediction = lstm_model.predict(lstm_input)[0][0]
    
    # Return predictions
    return render_template('index.html', 
                           result = res,
                           xgb_prediction=round(xgb_prediction, 2),
                           lstm_prediction=round(lstm_prediction, 2))

if __name__ == '__main__':
    app.run(host='0.0.0.0',port=5005,debug=True)
