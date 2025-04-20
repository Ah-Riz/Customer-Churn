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
    