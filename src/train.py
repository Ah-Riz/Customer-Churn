from preprocessing import preprocessing
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import cross_validate

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
        SVC(
            kernel='linear',
            C=10,
            gamma='scale',
            random_state=42
        )
    )

    pipe.fit(X_train, y_train)
    pipe.score(X_test, y_test)
    scoring = ['accuracy', 'f1_macro', 'roc_auc']
    cv_results = cross_validate(pipe, X_train, y_train, cv=5, scoring=scoring)

    print("Cross-validation results:")
    for metric in scoring:
        print(f"{metric}: {cv_results[f'test_{metric}'].mean():.4f} ± {cv_results[f'test_{metric}'].std():.4f}")

if __name__ == "__main__":
    main()