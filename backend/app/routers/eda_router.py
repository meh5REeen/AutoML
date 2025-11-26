from fastapi import APIRouter, UploadFile, File, HTTPException, Query
import os
import tempfile
from app.services.eda import EDAService, OutlierDetector
from app.utils.file_handler import load_csv

router = APIRouter()

@router.post("/analyze")
async def analyze_dataset(
    file: UploadFile = File(...),
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
    # Validate file type
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="File must be a CSV file")
    
    # Validate parameters
    if not 0 < test_size < 1:
        raise HTTPException(status_code=400, detail="test_size must be between 0 and 1")
    
    if zscore_threshold <= 0:
        raise HTTPException(status_code=400, detail="zscore_threshold must be positive")
    
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as tmp_file:
        contents = await file.read()
        tmp_file.write(contents)
        tmp_path = tmp_file.name
    
    try:
        # Load the CSV
        df = load_csv(tmp_path)
        
        # Validate that we have data
        if df.empty:
            raise HTTPException(status_code=400, detail="Uploaded CSV file is empty")
        
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
            "filename": file.filename,
            "data": report
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analyzing dataset: {str(e)}")
    
    finally:
        # Clean up temporary file
        if os.path.exists(tmp_path):
            os.remove(tmp_path)