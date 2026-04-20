import pandas as pd
from lightgbm import LGBMClassifier
import joblib
from src.preprocess import preprocess_data

def train_model():

    train = pd.read_csv("data/train.csv")

    y = train['Irrigation_Need']
    X = train.drop(['Irrigation_Need', 'id'], axis=1)

    X = preprocess_data(X)

    model = LGBMClassifier(
        n_estimators=800,
        learning_rate=0.03,
        max_depth=6,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        class_weight='balanced',
        random_state=42
    )

    model.fit(X, y)

    joblib.dump(model, "models/model.pkl")

    print("Model trained and saved!")

if __name__ == "__main__":
    train_model()