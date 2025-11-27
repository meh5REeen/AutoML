from fastapi import APIRouter, UploadFile, File, HTTPException, Query
import os
import tempfile
from app.services.eda import EDAService, OutlierDetector
from app.utils.file_handler import load_csv
import pandas as pd
import numpy as np
from app.session_manager import get_session_path, create_session
from charset_normalizer import from_path
import uuid

router = APIRouter()

@router.post("/analyze")
async def analyze_dataset(
    session_id:str,
    include_outliers: bool = Query(True, description="Include outlier detection analysis"),
    include_visualizations: bool = Query(True, description="Include plots and visualizations"),
    zscore_threshold: float = Query(3.0, description="Z-score threshold for outlier detection"),
    test_size: float = Query(0.2, description="Test set proportion for train/test split")
):
    """
    Upload a CSV file and perform comprehensive automated EDA in one go.
    
    Performs:
    1. Missing value analysis (per feature + global percent)
    2. Outlier detection (IQR method and Z-score method)
    3. Correlation matrix with heatmap visualization
    4. Distribution plots for numerical features
    5. Bar plots for categorical features
    6. Train/test split summary with visualization
    
    Query Parameters:
    - include_outliers: Include outlier detection (default: true)
    - include_visualizations: Generate plots as base64 PNG images (default: true)
    - zscore_threshold: Z-score threshold for outlier detection (default: 3.0)
    - test_size: Proportion of data for test split (default: 0.2)
    
    Returns:
        Complete EDA report with all analyses and visualizations (plots as base64 PNG images)
    """
    # get file from the session
    session_path = get_session_path(session_id)
    csv_path = os.path.join(session_path, "dataset.csv")
    excel_path = os.path.join(session_path, "dataset.xlsx")

    # Load dataset based on file type
    if os.path.exists(csv_path):
        # Auto-detect encoding for CSV
        try:
            df = pd.read_csv(csv_path, encoding="utf-8")
        except UnicodeDecodeError:
            detected = from_path(csv_path).best()
            encoding = detected.encoding if detected else "latin-1"
            df = pd.read_csv(csv_path, encoding=encoding)

    elif os.path.exists(excel_path):
        df = pd.read_excel(excel_path)

    else:
        raise HTTPException(404, "No dataset found for this session")
    # Validate parameters
    if not 0 < test_size < 1:
        raise HTTPException(status_code=400, detail="test_size must be between 0 and 1")
    
    if zscore_threshold <= 0:
        raise HTTPException(status_code=400, detail="zscore_threshold must be positive")
    
    
    try:
          
        # Generate comprehensive EDA report with all analyses and visualizations
        report = EDAService.generate_eda_report(
            df, 
            include_outliers=include_outliers,
            include_visualizations=include_visualizations,
            zscore_threshold=zscore_threshold,
            test_size=test_size
        )
        
        return {
            "status": "success",
            "data": report
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analyzing dataset: {str(e)}")