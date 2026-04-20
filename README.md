# 🌱 Irrigation Need Prediction

## 📌 Problem
Predict irrigation need (Low, Medium, High) using environmental and crop data.

## 📊 Dataset
- Features Soil Type, Crop Type, Temperature, Humidity, etc.
- Target Irrigation_Need
-📌 **Note:** Dataset not included due to size & licensing.  
👉 Download from Kaggle and place inside `data/` folder.

## ⚙️ Approach
- Data preprocessing (encoding categorical features)
- Handled class imbalance
- Feature engineering
- Model LightGBM

## 🧠 Model Details
- n_estimators = 800
- learning_rate = 0.03
- class_weight = balanced

## 📈 Results
- Kaggle Score 0.96797

## 🏗️ Project Structure
- Modular pipeline (train  predict)
- Saved model using joblib
- Clean code using src structure

## ▶️ How to Run

```bash
python -m src.train
python -m src.prediction
