from fastapi import APIRouter, Query
import os
from app.session_manager import get_session_path
from app.services.train import train_and_test_models,tune_models
from charset_normalizer import from_path
import pandas as pd
from fastapi import HTTPException
from typing import List

router = APIRouter()

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
        return {
            "Models": results,
            "Tuned-Models": results_tune
        }
    else:
        results = train_and_test_models(X, y, test_size=test_size, random_state=random_state, selected_models=models)
        return results