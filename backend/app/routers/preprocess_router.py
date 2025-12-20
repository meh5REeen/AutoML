from fastapi import APIRouter,HTTPException
from app.session_manager import get_session_path
import os
from charset_normalizer import from_path
import pandas as pd
from app.services.preprocess import handle_missing_values,handle_outliers,splitting_data,scale_numerical_features,encode_categorical_variables, remove_duplicates
from fastapi.encoders import jsonable_encoder
from app.services.issues import IssueDetector
import json
from datetime import datetime


router = APIRouter()

def save_clean_dataframe(df,session_id,original_file_name):
    folder_path = get_session_path(session_id)
    os.makedirs(folder_path, exist_ok=True)
    
    _,ext = os.path.splitext(original_file_name)
    ext = ext.lower()

    if ext == ".csv":
        output_path = os.path.join(folder_path, "data_cleaned.csv")
        df.to_csv(output_path,index=False)

    elif ext in [".xlsx",".xls"]:
        output_path = os.path.join(folder_path, "data_cleaned.xlsx")
        df.to_excel(output_path,index=False)

    else:
        raise ValueError("Unsupported file type for saving cleaned data")

    return output_path


def save_preprocessing_metadata(session_id, df_original, df_processed, preprocessing_params):
    """Save preprocessing metadata and detected issues"""
    folder_path = get_session_path(session_id)
    
    # Detect issues
    issue_detector = IssueDetector()
    issues_report = issue_detector.detect_all_issues(df_original, df_processed, preprocessing_params)
    
    # Create metadata
    metadata = {
        "timestamp": datetime.now().isoformat(),
        "parameters": preprocessing_params,
        "data_quality_before": {
            "total_rows": df_original.shape[0],
            "total_columns": df_original.shape[1],
            "missing_values": int(df_original.isnull().sum().sum()),
            "missing_percent": round((df_original.isnull().sum().sum() / (df_original.shape[0] * df_original.shape[1])) * 100, 2),
            "duplicate_rows": int(df_original.duplicated().sum()),
            "columns": list(df_original.columns)
        },
        "data_quality_after": {
            "total_rows": df_processed.shape[0],
            "total_columns": df_processed.shape[1],
            "missing_values": int(df_processed.isnull().sum().sum()),
            "missing_percent": round((df_processed.isnull().sum().sum() / (df_processed.shape[0] * df_processed.shape[1])) * 100, 2),
            "duplicate_rows": int(df_processed.duplicated().sum()),
            "columns": list(df_processed.columns)
        },
        "rows_removed": df_original.shape[0] - df_processed.shape[0],
        "issues_detected": issues_report
    }
    
    # Save metadata JSON
    metadata_path = os.path.join(folder_path, "preprocessing_metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2, default=str)
    
    return metadata_path





@router.post("/preprocess")
def preprocess_data(session_id: str,
                    missing_strategy: str = "Mean",
                    outlier_method: str = "Remove",
                    scaling_method: str = "Standard",
                    encoding_method: str = "OneHot",
                    test_size: float = 0.2,
                    target: str = None,
                    impute_constant=None):
    session_path = get_session_path(session_id)
    csv_path = os.path.join(session_path, "dataset.csv")
    excel_path = os.path.join(session_path, "dataset.xlsx")
    original_file_name = ''
    # Load dataset based on file type
    if os.path.exists(csv_path):
        # Auto-detect encoding for CSV
        try:
            df = pd.read_csv(csv_path, encoding="utf-8")
            original_file_name = 'dataset.csv'
        except UnicodeDecodeError:
            detected = from_path(csv_path).best()
            encoding = detected.encoding if detected else "latin-1"
            df = pd.read_csv(csv_path, encoding=encoding)
            original_file_name = 'dataset.csv'

    elif os.path.exists(excel_path):
        df = pd.read_excel(excel_path)

    else:
        raise HTTPException(404, "No dataset found for this session")

    df = handle_missing_values(strategy=missing_strategy,df=df,fill_value=impute_constant)
    df = handle_outliers(df=df,method=outlier_method)
    
    target_series = df[target]  # store original target
    df_before_encoding = df.copy()  # Keep copy for issue detection
    
    df = remove_duplicates(df=df)
    df = df.drop(columns=[target]) 
    df = encode_categorical_variables(df=df,encoding_type=encoding_method)

    df = scale_numerical_features(df=df,scaling_type=scaling_method)
    df[target] = target_series
    X_train,X_test,y_train,y_test = splitting_data(df=df,target=target,test_size=test_size)

    cleaned_data_path = save_clean_dataframe(df=df,session_id=session_id,original_file_name=original_file_name)
    
    # Save preprocessing metadata
    preprocessing_params = {
        "missing_strategy": missing_strategy,
        "outlier_method": outlier_method,
        "scaling_method": scaling_method,
        "encoding_method": encoding_method,
        "test_size": test_size,
        "impute_constant": impute_constant
    }
    metadata_path = save_preprocessing_metadata(session_id, df_before_encoding, df, preprocessing_params)

    return jsonable_encoder({
        "Splitted_data": {
            "X_train": X_train.to_dict(orient="records"),
            "X_test": X_test.to_dict(orient="records"),
            "y_train": y_train.tolist(),
            "y_test": y_test.tolist()
        },
        "cleaned_path": cleaned_data_path,
        "metadata_path": metadata_path
    })
