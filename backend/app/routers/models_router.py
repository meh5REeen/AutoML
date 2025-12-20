from fastapi import APIRouter, Query
import os
from app.session_manager import get_session_path
from app.services.train import train_and_test_models,tune_models
from charset_normalizer import from_path
import pandas as pd
from fastapi import HTTPException
from typing import List
import json
from datetime import datetime

router = APIRouter()

def save_model_results(session_id, results, tuned_results, hyperparams):
    """Save model training results to session"""
    folder_path = get_session_path(session_id)
    
    # Find best model
    best_model = None
    best_score = -1
    best_reason = ""
    
    all_results = {}
    
    # Add regular models
    if results:
        all_results.update(results)
    
    # Add tuned models
    if tuned_results:
        all_results.update(tuned_results)
    
    # Find best based on F1 score (or accuracy if F1 not available)
    for model_name, metrics in all_results.items():
        score = metrics.get('f1_score', metrics.get('accuracy', 0))
        if score > best_score:
            best_score = score
            best_model = model_name
            best_reason = f"Best F1 score: {score:.4f}" if 'f1_score' in metrics else f"Best accuracy: {score:.4f}"
    
    model_results = {
        "timestamp": datetime.now().isoformat(),
        "hyperparameters": hyperparams,
        "models": []
    }
    
    # Add all models to results
    for model_name, metrics in all_results.items():
        model_entry = {
            "name": model_name,
            "metrics": metrics
        }
        model_results["models"].append(model_entry)
    
    model_results["best_model"] = {
        "name": best_model,
        "reason": best_reason,
        "f1_score": best_score
    }
    
    # Save model results JSON
    model_results_path = os.path.join(folder_path, "model_results.json")
    with open(model_results_path, 'w') as f:
        json.dump(model_results, f, indent=2, default=str)
    
    return model_results_path

@router.get("/models")
def compare_models(
    session_id: str,
    target: str = None,
    test_size: float = Query(0.2, ge=0.1, le=0.9),
    random_state: int = 42,
    optimize: bool = False,
    models: List[str] = Query(
        [
            "Logistic Regression",
            "K-Neighbors Classifier",
            "Decision Tree Classifier",
            "Gaussian Naive Bayes",
            "Random Forest",
            "Support Vector Machine",
            "Decision Tree Rule-based"
        ],
        description="List of model names to train"
    )
):
    session_path = get_session_path(session_id)
    csv_path = os.path.join(session_path, "data_cleaned.csv")
    excel_path = os.path.join(session_path, "data_cleaned.xlsx")
    original_file_name = ''
    # Load dataset based on file type
    if os.path.exists(csv_path):
        # Auto-detect encoding for CSV
        try:
            df = pd.read_csv(csv_path, encoding="utf-8")
            original_file_name = 'data_cleaned.csv'
        except UnicodeDecodeError:
            detected = from_path(csv_path).best()
            encoding = detected.encoding if detected else "latin-1"
            df = pd.read_csv(csv_path, encoding=encoding)
            original_file_name = 'data_cleaned.csv'

    elif os.path.exists(excel_path):
        df = pd.read_excel(excel_path)

    else:
        raise HTTPException(404, "No dataset found for this session")
    print("Dataset loaded for model comparison.")
    if target not in df.columns:
        raise HTTPException(400, "Target column 'target' not found in dataset")
    X = df.drop(columns=[target])
    y = df[target]
    
    # Validate selected models
    valid_models = [
        "Logistic Regression",
        "K-Neighbors Classifier",
        "Decision Tree Classifier",
        "Gaussian Naive Bayes",
        "Random Forest",
        "Support Vector Machine",
        "Decision Tree Rule-based"
    ]
    invalid_models = [m for m in models if m not in valid_models]
    if invalid_models:
        raise HTTPException(
            400,
            f"Invalid model names: {invalid_models}. Valid models are: {valid_models}"
        )
    
    if optimize:
        results_tune = tune_models(X, y, test_size=test_size, random_state=random_state, selected_models=models)
        results = train_and_test_models(X, y, test_size=test_size, random_state=random_state, selected_models=models)
        
        hyperparams = {
            "test_size": test_size,
            "random_state": random_state,
            "optimize": optimize
        }
        
        # Save model results
        model_results_path = save_model_results(session_id, results, results_tune, hyperparams)
        
        return {
            "Models": results,
            "Tuned-Models": results_tune,
            "model_results_path": model_results_path
        }
    else:
        results = train_and_test_models(X, y, test_size=test_size, random_state=random_state, selected_models=models)
        
        hyperparams = {
            "test_size": test_size,
            "random_state": random_state,
            "optimize": optimize
        }
        
        # Save model results
        model_results_path = save_model_results(session_id, results, None, hyperparams)
        
        return {
            "models": results,
            "model_results_path": model_results_path
        }