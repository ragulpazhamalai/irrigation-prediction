import pandas as pd
import joblib
from src.preprocess import preprocess_data

def generate_submission():

    test = pd.read_csv("data/test.csv")
    ids = test['id']

    X_test = test.drop('id', axis=1)
    X_test = preprocess_data(X_test)

    model = joblib.load("models/model.pkl")

    preds = model.predict(X_test)

    submission = pd.DataFrame({
        "id": ids,
        "Irrigation_Need": preds
    })

    submission.to_csv("outputs/submission.csv", index=False)

    print("Submission file created!")

if __name__ == "__main__":
    generate_submission()