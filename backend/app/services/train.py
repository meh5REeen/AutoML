from app.services.preprocess import splitting_data
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score,roc_auc_score , confusion_matrix
import pandas as pd
import numpy as np
import time


def get_models(random_state):
    models = {
        "Logistic Regression": LogisticRegression(),
        "K-Neighbors Classifier": KNeighborsClassifier(),
        "Decision Tree Classifier": DecisionTreeClassifier(),
        "Gaussian Naive Bayes": GaussianNB(),
        "Random Forest": RandomForestClassifier(random_state=random_state),
        "Support Vector Machine": SVC(probability=True, random_state=random_state),
        "Decision Tree Rule-based": DecisionTreeClassifier(max_depth=5, random_state=random_state)
    }
    return models

def roc_auc(y_true,model,X_test):
    if not hasattr(model,"predict_proba"):
        return None
    y_probs = model.predict_proba(X_test)

    # unique classes
    classes = np.unique(y_true)
    if len(classes) == 2:
        return roc_auc_score(y_true, y_probs[:, 1])
    else:
        return roc_auc_score(y_true, y_probs, multi_class='ovr')
    

def parameter_grid():
    param_grid = {
        "Logistic Regression": {
            "C": [0.01, 0.1, 1, 10],
            "solver": ["lbfgs"],
            "max_iter": [100, 200, 300],
            "penalty":['l2']
        },
        "K-Neighbors Classifier": {
            "n_neighbors": [3, 5, 7, 9],
            "weights": ["uniform", "distance"],
            "metric": ["euclidean", "manhattan"]
        },
        "Decision Tree Classifier": {
            "max_depth": [None, 5, 10, 15],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
            "criterion": ["gini", "entropy"]
        },
        "Gaussian Naive Bayes": {
            "var_smoothing": np.logspace(-9, -1, 10)
        },
        "Random Forest": {
            "n_estimators": [50, 100, 200],
            "max_depth": [None, 10, 20, 30],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
            "criterion": ["gini", "entropy"]
        },
        "Support Vector Machine": {
            "C": [0.1, 1, 10, 100],
            "kernel": ["linear", "rbf", "poly"],
            "gamma": ["scale", "auto"]
        },
        "Decision Tree Rule-based": {
            "max_depth": [3, 5, 7, 10],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
            "criterion": ["gini", "entropy"]
        }
    }
    return param_grid



def train_and_test_models(X, y, test_size: float, random_state=42, selected_models=None):

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    all_models = get_models(random_state)
    
    # Filter models if selected_models is provided
    if selected_models:
        models = {name: model for name, model in all_models.items() if name in selected_models}
    else:
        models = all_models
    
    results = {}
    for model_name, model in models.items():
        start_time = time.time()
        model.fit(X_train, y_train)
        end_time = time.time()

        training_time = end_time - start_time
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        roc = roc_auc(y_test,model,X_test)
        cm_df = pd.DataFrame(cm, index=range(1, len(cm)+1), columns=range(1, len(cm)+1))
        if roc is not None:
            results[model_name] ={
                 "accuracy": accuracy_score(y_test, y_pred),
                    "precision": precision_score(y_test, y_pred, average='weighted', zero_division=0),
                    "recall": recall_score(y_test, y_pred, average='weighted', zero_division=0),
                    "f1_score": f1_score(y_test, y_pred, average='weighted', zero_division=0),
                
                "roc_auc":roc,
                "confusion_matrix": cm_df.values.tolist(),
                "training_time": training_time

            }
            
        else:
            results[model_name] ={
                 "accuracy": accuracy_score(y_test, y_pred),
                    "precision": precision_score(y_test, y_pred, average='weighted', zero_division=0),
                    "recall": recall_score(y_test, y_pred, average='weighted', zero_division=0),
                    "f1_score": f1_score(y_test, y_pred, average='weighted', zero_division=0),
                
                "training_time": training_time,
                "confusion_matrix": cm_df.values.tolist(),
            }


    return results

def tune_models(X, y, test_size: float, random_state=42, selected_models=None):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    all_models = get_models(random_state)
    
    # Filter models if selected_models is provided
    if selected_models:
        models = {name: model for name, model in all_models.items() if name in selected_models}
    else:
        models = all_models
    
    results_tuned = {}
    for model_name, model in models.items():
        grid_params = parameter_grid().get(model_name, {})
        if grid_params:
            grid = GridSearchCV(estimator=model, param_grid=grid_params, cv=5, n_jobs=-1)
            start_time = time.time()
            grid.fit(X_train, y_train)
            end_time = time.time()
            tuning_time = end_time - start_time
            best_model = grid.best_estimator_
            best_params = grid.best_params_
            y_pred = best_model.predict(X_test)
            cm = confusion_matrix(y_test, y_pred)
            cm_df = pd.DataFrame(cm, index=range(1, len(cm)+1), columns=range(1, len(cm)+1))

            roc = roc_auc(y_test,best_model,X_test)
            results_tuned={}
            if roc is not None:
                results_tuned[model_name + " (Tuned)"] = {
                    "accuracy": accuracy_score(y_test, y_pred),
                    "precision": precision_score(y_test, y_pred, average='weighted', zero_division=0),
                    "recall": recall_score(y_test, y_pred, average='weighted', zero_division=0),
                    "f1_score": f1_score(y_test, y_pred, average='weighted', zero_division=0),                   "roc_auc": roc,
                    "confusion_matrix": cm_df.values.tolist(),
                    "training_time": tuning_time,
                    "best_params": best_params
                }
            else:
                results_tuned[model_name + " (Tuned)"] = {
                    "accuracy": accuracy_score(y_test, y_pred),
                    "precision": precision_score(y_test, y_pred, average='weighted', zero_division=0),
                    "recall": recall_score(y_test, y_pred, average='weighted', zero_division=0),
                    "f1_score": f1_score(y_test, y_pred, average='weighted', zero_division=0),
                    "training_time": tuning_time,
                    "confusion_matrix": cm_df.values.tolist(),
                    "best_params": best_params
                }

    return results_tuned