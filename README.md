# Customer-Churn

This project provides a RESTful API for predicting customer churn using a machine learning model. The API is built with **FastAPI** and allows users to send customer data and receive a prediction indicating whether the customer is likely to churn.

---

## Features

- **Prediction Endpoint**: Accepts customer data and returns a churn prediction (`yes` or `no`).
- **Preprocessing**: Automatically preprocesses input data to match the model's requirements.
- **Model Integration**: Uses a pre-trained machine learning model for predictions.

---

## Requirements

- Python 3.8 or higher
- FastAPI
- Uvicorn
- Joblib
- Pandas
- Scikit-learn

---

## Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-repo/customer-churn-api.git
   cd customer-churn-api
   ```

2. **Set Up a Virtual Environtment**:
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3. **Install Dependencies**:
    ```bash
    pip install -r requirement.txt
    ```

4. **Ensure Directory Structure**:
    ```bash
    project/
    ├── app/
    │   ├── main.py
    ├── src/
    │   ├── preprocessing.py
    ├── models/
    │   ├── model.pkl
    ├── data/
    │   ├── mappings.json
    │   ├── columns.json
    ├── requirements.txt
    └── [README.md]
    ```

## Usage

1. **Start API Server**:
    ```bash
    uvicorn app.main:app --host 0.0.0.0 --port 8000
    ```

2. **Access the API**:
    - Open your browser.
    - Root endpoint: http://127.0.0.1:8000/

3. **Make a Prediction**:
    - Send a #POST# request to http://127.0.0.1:8000/docs#/default/predict_predict__post with the required customer data.
    **Example Request**:
    ```bash
    curl -X POST "http://127.0.0.1:8000/docs#/default/predict_predict__post" \
    -H "Content-Type: application/json" \
    -d '[
        "Male", 0, "No", "Yes", 62, "Yes", "No", "DSL", "Yes", "Yes", "No", "No", "No", "No",
        "One year", "No", "Bank transfer (automatic)", 56.15, 3487.95
    ]'
    ```
    **Example Response**:
    ```bash
    {
        "prediction": "no"
    }
    ```