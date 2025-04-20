from fastapi import FastAPI, requests
import joblib
import os
import json
import sys
sys.path.append(os.path.abspath("src"))
from preprocessing import preprocess_input

app = FastAPI()

model = joblib.load(os.path.join(os.path.abspath("models"), "model.pkl"))
with open("data/mappings.json", "r") as f:
    label_mapping = json.load(f)

@app.post("/predict/")
async def predict(gender : str, SeniorCitizen : int, Partner : str, Dependents : str, tenure : int, PhoneService : str, MultipleLines : str, InternetService : str, OnlineSecurity : str, OnlineBackup : str, DeviceProtection : str, TechSupport : str, StreamingTV : str, StreamingMovies : str, Contract : str, PaperlessBilling : str, PaymentMethod : str, MonthlyCharges : float, TotalCharges : float):
    try:
        data = [gender, SeniorCitizen, Partner, Dependents, tenure, PhoneService, MultipleLines, InternetService, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies, Contract, PaperlessBilling, PaymentMethod, MonthlyCharges, TotalCharges]
        data = preprocess_input(data)
        
        # Make predictions
        prediction = model.predict(data)
        
        reversed_mapping = {v: k for k, v in label_mapping["Churn"].items()}
        churn_label = reversed_mapping[prediction[0]]
        
        return {"prediction": churn_label}
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
