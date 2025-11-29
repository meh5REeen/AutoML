from fastapi import APIRouter
import os
from app.session_manager import get_session_path
from app.services.train import train_and_test_models,tune_models
from charset_normalizer import from_path
import pandas as pd
from fastapi import HTTPException

router = APIRouter()

@router.get("/models")
def compare_models(session_id:str,test_size:float,target,random_state:int,optimize:bool=False):
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
    if optimize:
        results_tune = tune_models(X,y,test_size=test_size,random_state=random_state)
        results = train_and_test_models(X,y,test_size=test_size,random_state=random_state)
        return {
            "Models": results,
            "Tuned-Models": results_tune
        }
    else:
        results = train_and_test_models(X,y,test_size=test_size,random_state=random_state)
        return results