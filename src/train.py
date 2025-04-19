from preprocessing import preprocessing
import sklearn
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, accuracy_score, f1_score
import joblib
import os

sklearn.set_config(enable_metadata_routing=True)

def training():
    path = "data/WA_Fn-UseC_-Telco-Customer-Churn.csv"

    train_data, val_data = preprocessing(path)
    X = train_data[:, :-1]
    y = train_data[:, -1].astype(int)
    X_val = val_data[:, :-1]
    y_val = val_data[:, -1].astype(int)

    pipe = make_pipeline(
        StandardScaler(),
        SVC()
    )
    param_grid = {
        'svc__C': [0.1, 1, 10],
        'svc__kernel': ['linear', 'rbf'],
        'svc__gamma': ['scale', 'auto']
    }
    
    f1_scorer = make_scorer(f1_score, average='binary').set_score_request(sample_weight=True)
    accuracy_scorer = make_scorer(accuracy_score).set_score_request(sample_weight=True)
    scoring = {"accuracy":accuracy_scorer, "f1":f1_scorer}
    # scoring = {"f1":f1_scorer}

    grid = GridSearchCV(pipe, param_grid, cv=5, n_jobs=-1, refit='f1', scoring=scoring, return_train_score=True)
    grid.fit(X, y)
    
    print(f"Best parameters: {grid.best_params_}")
    print(f"Best F1 score (best cross-validation): {grid.best_score_:.4f}")

    mean_accuracy = grid.cv_results_["mean_test_accuracy"]
    mean_f1 = grid.cv_results_["mean_test_f1"]

    print(f"Mean accuracy (Mean cross-validation): {mean_accuracy.mean():.4f}")
    print(f"Mean F1 score (Mean cross-validation): {mean_f1.mean():.4f}")

    val_predictions = grid.predict(X_val)
    print(f"Accuracy validation: {accuracy_score(y_val, val_predictions):.4f}")
    print(f"F1 score validation: {f1_score(y_val, val_predictions):.4f}")

    joblib.dump(grid, os.path.join(os.path.abspath("models"),"model.pkl"))
    
    return True

if __name__ == "__main__":
    training()