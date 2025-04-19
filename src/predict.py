import joblib
import os
import json
from preprocessing import preprocess_input

def predict(data):
    data = preprocess_input(data)
    
    # Load the model
    model_path = os.path.join(os.path.abspath("models"), "model.pkl")
    model = joblib.load(model_path)

    # Make predictions
    prediction = model.predict(data)
    
    with open("data/mappings.json", "r") as f:
        label_mapping = json.load(f)
    reversed_mapping = {v: k for k, v in label_mapping["Churn"].items()}
    churn_label = reversed_mapping[prediction[0]]
    
    print(f"Prediction: {churn_label}")


if __name__ == "__main__":
    data = ["Male", 0, "No", "Yes", 62, "Yes", "No", "DSL", "Yes", "Yes", "No", "No", "No", "No", "One year", "No", "Bank transfer (automatic)", 56.15, 3487.95]

    predict(data)