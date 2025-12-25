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

    # Store original row count
    original_row_count = df.shape[0]
    print(f"[PREPROCESS] Starting with {original_row_count} rows")

    try:
        print(f"[PREPROCESS] Handling missing values with strategy: {missing_strategy}")
        df = handle_missing_values(strategy=missing_strategy,df=df,fill_value=impute_constant)
        print(f"[PREPROCESS] After missing values: {df.shape[0]} rows")
    except Exception as e:
        print(f"[ERROR] Failed in handle_missing_values: {str(e)}")
        raise HTTPException(500, f"Error handling missing values: {str(e)}")
    
    try:
        print(f"[PREPROCESS] Handling outliers with method: {outlier_method}")
        df = handle_outliers(df=df,method=outlier_method)
        print(f"[PREPROCESS] After outliers: {df.shape[0]} rows")
    except Exception as e:
        print(f"[ERROR] Failed in handle_outliers: {str(e)}")
        raise HTTPException(500, f"Error handling outliers: {str(e)}")
    
    try:
        print(f"[PREPROCESS] Storing target column: {target}")
        target_series = df[target]  # store original target
        df_before_encoding = df.copy()  # Keep copy for issue detection
        print(f"[PREPROCESS] Target column stored successfully")
    except Exception as e:
        print(f"[ERROR] Failed to store target: {str(e)}")
        raise HTTPException(500, f"Error accessing target column '{target}': {str(e)}")
    
    try:
        print(f"[PREPROCESS] Removing duplicates")
        df = remove_duplicates(df=df)
        print(f"[PREPROCESS] After duplicates: {df.shape[0]} rows")
    except Exception as e:
        print(f"[ERROR] Failed in remove_duplicates: {str(e)}")
        raise HTTPException(500, f"Error removing duplicates: {str(e)}")
    
    try:
        print(f"[PREPROCESS] Dropping target column for encoding")
        df = df.drop(columns=[target])
        print(f"[PREPROCESS] Encoding categorical variables with method: {encoding_method}")
        df, categorical_mappings = encode_categorical_variables(df=df,encoding_type=encoding_method)
        print(f"[PREPROCESS] After encoding: {df.shape}")
        
        # Save categorical mappings for prediction UI
        if categorical_mappings:
            mappings_path = os.path.join(session_path, "categorical_mappings.json")
            with open(mappings_path, 'w') as f:
                json.dump(categorical_mappings, f, indent=2)
            print(f"[PREPROCESS] Saved categorical mappings to {mappings_path}")
    except Exception as e:
        print(f"[ERROR] Failed in encode_categorical_variables: {str(e)}")
        raise HTTPException(500, f"Error encoding categorical variables: {str(e)}")

    try:
        print(f"[PREPROCESS] Scaling numerical features with method: {scaling_method}")
        df = scale_numerical_features(df=df,scaling_type=scaling_method)
        print(f"[PREPROCESS] After scaling: {df.shape}")
    except Exception as e:
        print(f"[ERROR] Failed in scale_numerical_features: {str(e)}")
        raise HTTPException(500, f"Error scaling numerical features: {str(e)}")
    
    try:
        print(f"[PREPROCESS] Adding target column back")
        df[target] = target_series
        print(f"[PREPROCESS] Splitting data with test_size: {test_size}")
        X_train,X_test,y_train,y_test = splitting_data(df=df,target=target,test_size=test_size)
        print(f"[PREPROCESS] Split complete - Train: {len(X_train)}, Test: {len(X_test)}")
    except Exception as e:
        print(f"[ERROR] Failed in splitting_data: {str(e)}")
        raise HTTPException(500, f"Error splitting data: {str(e)}")

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

    # Calculate rows removed
    rows_removed = original_row_count - df.shape[0]

    return jsonable_encoder({
        "Splitted_data": {
            "X_train": X_train.head(5).to_dict(orient="records"),  # Only send preview
            "X_test": X_test.head(5).to_dict(orient="records"),    # Only send preview
            "y_train": y_train.tolist()[:5],  # Only send preview
            "y_test": y_test.tolist()[:5],    # Only send preview
            "train_count": len(X_train),  # Actual count
            "test_count": len(X_test)     # Actual count
        },
        "cleaned_path": cleaned_data_path,
        "metadata_path": metadata_path,
        "rows_removed": rows_removed,
        "original_rows": original_row_count,
        "final_rows": df.shape[0]
    })
