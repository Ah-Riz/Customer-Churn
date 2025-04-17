from preprocessing import preprocessing
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import cross_validate
import joblib
import os

def main():
    path = "data/WA_Fn-UseC_-Telco-Customer-Churn.csv"
    gender_list = {"female":0, "male":1}
    MultipleLines_list = {"yes": 1, "no": 0, "no phone service": -1}
    service_installation = {"yes": 1, "no": 0, "no internet service": -1}
    InternetService_list = {"dsl": 1, "fiber optic": 2, "no": 0}
    Contract_list = {"month-to-month": 0, "one year": 1, "two year": 2}
    paymentMethod_list = {"bank transfer (automatic)": 0, "credit card (automatic)": 1, "electronic check": 2, "mailed check": 3}

    data = preprocessing(path, gender_list, MultipleLines_list, InternetService_list, service_installation, Contract_list, paymentMethod_list)
    data = data.to_numpy()[:, 1:]

    X = data[:, 1:-1]
    y = data[:, -1].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    pipe = make_pipeline(
        StandardScaler(),
        SVC()
    )
    param_grid = {
        'svc__C': [0.1, 1, 10],
        'svc__kernel': ['linear', 'rbf'],
        'svc__gamma': ['scale', 'auto']
    }

    scoring = ['accuracy', 'f1_macro']
    grid = GridSearchCV(pipe, param_grid, cv=5, scoring=scoring, n_jobs=-1, refit='f1_macro', return_train_score=True)
    grid.fit(X_train, y_train)
    grid.score(X_test, y_test)
    print(f"Best f1_macro: {grid.best_score_:.4f}")
    print(f"Best index: {grid.best_index_}")
    print(f"Best parameters: {grid.best_params_}")

    joblib.dump(grid, os.path.join(os.path.abspath("models"),"model.pkl"))

if __name__ == "__main__":
    main()