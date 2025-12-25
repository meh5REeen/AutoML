from fastapi import APIRouter, HTTPException, UploadFile, File, Form
import os
import uuid
import pandas as pd
from app.session_manager import get_session_path, create_session
from charset_normalizer import from_path
import numpy as np

router = APIRouter()


def ensure_valid_session_id(session_id: str | None) -> str:
    """
    Ensure the provided session_id is valid. If not, create a new session and return its ID.
    """
    if not session_id:
        session_id = create_session() 
        return session_id
    try: 
        uuid_obj = uuid.UUID(session_id, version=4) 
        return session_id # valid, return as-is
    except ValueError: 
        session_id = create_session() 
        return session_id

# @router.post('/upload')
# async def upload_dataset(
#     file: UploadFile = File(...),
#     session_id: str | None = Form(None)
# ):
#     """
#     Upload a dataset file (CSV, XLS, XLSX).
#     """
#     session_id = ensure_valid_session_id(session_id)
#     session_path = get_session_path(session_id)

#     filename = file.filename.lower()
#     print("Uploaded filename:", filename)
#     # Determine file extension
#     if filename.endswith(".csv"):
#         ext = ".csv"
#     elif filename.endswith(".xlsx") or filename.endswith(".xls"):
#         ext = ".xlsx"
#     else:
#         raise HTTPException(
#             status_code=400,
#             detail="Unsupported file type. Only CSV, XLS, XLSX allowed."
#         )

#     dataset_path = os.path.join(session_path, f"dataset{ext}")

#     # Save file exactly as uploaded 
#     with open(dataset_path, "wb") as f:
#         f.write(await file.read())


#     return {
#         "message": f"Dataset uploaded successfully as {ext}",
#         "path": dataset_path,
#         "session_id": session_id,
        
#     }



# @router.get('/metadata')
# def get_metadata(session_id: str):
#     """
#     Extract metadata from the uploaded dataset.
#     """
#     session_path = get_session_path(session_id)
#     csv_path = os.path.join(session_path, "dataset.csv")
#     excel_path = os.path.join(session_path, "dataset.xlsx")

#     # Load dataset based on file type
#     if os.path.exists(csv_path):
#         # Auto-detect encoding for CSV
#         try:
#             df = pd.read_csv(csv_path, encoding="utf-8")
#         except UnicodeDecodeError:
#             detected = from_path(csv_path).best()
#             encoding = detected.encoding if detected else "latin-1"
#             df = pd.read_csv(csv_path, encoding=encoding)

#     elif os.path.exists(excel_path):
#         df = pd.read_excel(excel_path)

#     else:
#         raise HTTPException(404, "No dataset found for this session")

#     # Metadata
#     rows, cols = df.shape
#     dtypes = df.dtypes.apply(lambda x: x.name)
#     summary_stats = df.describe(include="all")
#     summary_stats = summary_stats.replace([np.nan, np.inf, -np.inf], None).to_dict()


#     # Identify target column
#     target_column = None
#     class_distribution = {}

#     if df.dtypes[-1] in ["object", "category"]:
#         target_column = df.columns[-1]
#         class_distribution = df[target_column].value_counts().to_dict()

#     return {
#         "rows": rows,
#         "columns": cols,
#         "column_types": dtypes,
#         "summary_statistics": summary_stats,
#         "target_column": target_column,
#         "class_distribution": class_distribution
#     }
@router.post('/upload')
async def upload_dataset(
    file: UploadFile = File(...),
    session_id: str | None = Form(None)
):
    """
    Upload a dataset file (CSV, XLS, XLSX) and return its metadata in one call.
    """
    session_id = ensure_valid_session_id(session_id)
    session_path = get_session_path(session_id)

    original_filename = file.filename
    filename_lower = original_filename.lower()

    # Determine file extension
    if filename_lower.endswith(".csv"):
        ext = ".csv"
    elif filename_lower.endswith(".xlsx") or filename_lower.endswith(".xls"):
        ext = ".xlsx"
    else:
        raise HTTPException(
            status_code=400,
            detail="Unsupported file type. Only CSV, XLS, XLSX allowed."
        )

    # Save the file internally
    # Ensure session directory exists
    os.makedirs(session_path, exist_ok=True)
    
    dataset_path = os.path.join(session_path, f"dataset{ext}")
    
    with open(dataset_path, "wb") as f:
        f.write(await file.read())

    # Load dataset for metadata
    if ext == ".csv":
        try:
            df = pd.read_csv(dataset_path, encoding="utf-8")
        except UnicodeDecodeError:
            detected = from_path(dataset_path).best()
            encoding = detected.encoding if detected else "latin-1"
            df = pd.read_csv(dataset_path, encoding=encoding)
    else:
        df = pd.read_excel(dataset_path)

    # Metadata extraction
    columns = df.columns.tolist()
    dtypes = df.dtypes.apply(lambda x: x.name).to_dict()

    # Auto-detect target column (suggest last categorical column, but user can select any)
    target_col = None
    class_distribution = {}
    
    # Try to auto-detect categorical target (last categorical column)
    for col in reversed(df.columns):
        if df[col].dtype in ["object", "category"]:
            target_col = col
            class_distribution = df[col].value_counts().to_dict()
            break
    
    # If no categorical column found, suggest the last column
    if not target_col and len(columns) > 0:
        target_col = columns[-1]
        # For numeric columns, show value distribution if it has few unique values
        if df[target_col].nunique() <= 10:
            class_distribution = df[target_col].value_counts().to_dict()

    return {
        "message": f"Dataset uploaded successfully as {ext}",
        "session_id": session_id,
        "fileName": original_filename,
        "columns": columns,
        "rows": df.shape[0],
        "dtypes": dtypes,
        "targetCol": target_col,
        "classDistribution": class_distribution,
        "preview": df.head(5).to_dict(orient="records")
    }