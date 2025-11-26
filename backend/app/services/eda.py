import pandas as pd
import numpy as np
from typing import Dict, Any, List
from scipy import stats
from app.utils.plot_generator import PlotGenerator


class OutlierDetector:
    """Detect outliers using IQR and Z-score methods."""
    
    @staticmethod
    def iqr_method(series: pd.Series) -> Dict[str, Any]:
        """
        Detect outliers using Interquartile Range (IQR) method.
        
        Returns:
            - outlier_indices: Indices of outlier values
            - outlier_values: The actual outlier values
            - outlier_count: Number of outliers
            - outlier_percent: Percentage of outliers
            - bounds: Lower and upper bounds
        """
        if not pd.api.types.is_numeric_dtype(series):
            return None
        
        series_clean = series.dropna()
        if len(series_clean) == 0:
            return None
        
        Q1 = series_clean.quantile(0.25)
        Q3 = series_clean.quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outlier_mask = (series_clean < lower_bound) | (series_clean > upper_bound)
        outlier_indices = series_clean[outlier_mask].index.tolist()
        outlier_values = series_clean[outlier_mask].tolist()
        outlier_count = outlier_mask.sum()
        outlier_percent = (outlier_count / len(series_clean)) * 100 if len(series_clean) > 0 else 0
        
        return {
            "method": "IQR",
            "outlier_count": int(outlier_count),
            "outlier_percent": round(outlier_percent, 2),
            "lower_bound": float(lower_bound),
            "upper_bound": float(upper_bound),
            "outlier_indices": outlier_indices[:100],  # Limit to first 100
            "outlier_values": [float(v) for v in outlier_values[:100]]
        }
    
    @staticmethod
    def zscore_method(series: pd.Series, threshold: float = 3.0) -> Dict[str, Any]:
        """
        Detect outliers using Z-score method.
        
        Parameters:
            - threshold: Z-score threshold (default 3.0, typical values: 2.5-3.0)
        
        Returns:
            - outlier_indices: Indices of outlier values
            - outlier_values: The actual outlier values
            - outlier_count: Number of outliers
            - outlier_percent: Percentage of outliers
            - threshold: Z-score threshold used
        """
        if not pd.api.types.is_numeric_dtype(series):
            return None
        
        series_clean = series.dropna()
        if len(series_clean) == 0:
            return None
        
        z_scores = np.abs(stats.zscore(series_clean))
        outlier_mask = z_scores > threshold
        
        outlier_indices = series_clean[outlier_mask].index.tolist()
        outlier_values = series_clean[outlier_mask].tolist()
        outlier_count = outlier_mask.sum()
        outlier_percent = (outlier_count / len(series_clean)) * 100 if len(series_clean) > 0 else 0
        
        return {
            "method": "Z-Score",
            "threshold": threshold,
            "outlier_count": int(outlier_count),
            "outlier_percent": round(outlier_percent, 2),
            "outlier_indices": outlier_indices[:100],  # Limit to first 100
            "outlier_values": [float(v) for v in outlier_values[:100]]
        }
    
    @staticmethod
    def analyze(df: pd.DataFrame, zscore_threshold: float = 3.0) -> Dict[str, Any]:
        """
        Detect outliers in all numeric columns using both methods.
        
        Returns:
            Dictionary with outlier detection results for each numeric column
        """
        numeric_columns = df.select_dtypes(include=['number']).columns.tolist()
        
        outliers_analysis = {}
        for column in numeric_columns:
            iqr_result = OutlierDetector.iqr_method(df[column])
            zscore_result = OutlierDetector.zscore_method(df[column], threshold=zscore_threshold)
            
            if iqr_result or zscore_result:
                outliers_analysis[column] = {
                    "iqr": iqr_result,
                    "zscore": zscore_result
                }
        
        return outliers_analysis


class MissingValueAnalyzer:
    """Analyze missing values in a dataset."""
    
    @staticmethod
    def analyze(df: pd.DataFrame) -> Dict[str, Any]:
        """
        Perform comprehensive missing value analysis.
        
        Returns:
            Dictionary containing:
            - per_feature: Missing values per column
            - global_percent: Global missing value percentage
            - total_missing: Total missing values
            - total_cells: Total cells in dataset
        """
        total_cells = df.shape[0] * df.shape[1]
        total_missing = df.isnull().sum().sum()
        global_percent = (total_missing / total_cells) * 100 if total_cells > 0 else 0
        
        per_feature = {}
        for column in df.columns:
            missing_count = df[column].isnull().sum()
            missing_percent = (missing_count / len(df)) * 100 if len(df) > 0 else 0
            per_feature[column] = {
                "count": int(missing_count),
                "percent": round(missing_percent, 2),
                "non_missing": int(len(df) - missing_count)
            }
        
        return {
            "per_feature": per_feature,
            "global_percent": round(global_percent, 2),
            "total_missing": int(total_missing),
            "total_cells": int(total_cells),
            "total_rows": df.shape[0],
            "total_columns": df.shape[1]
        }


class EDAService:
    """Main EDA service for automated analysis."""
    
    @staticmethod
    def generate_eda_report(df: pd.DataFrame, include_outliers: bool = True, 
                           include_visualizations: bool = True,
                           zscore_threshold: float = 3.0,
                           test_size: float = 0.2) -> Dict[str, Any]:
        """
        Generate comprehensive EDA report with all analyses and visualizations.
        
        Parameters:
            - include_outliers: Whether to include outlier detection
            - include_visualizations: Whether to generate plots (as base64 images)
            - zscore_threshold: Z-score threshold for outlier detection (default 3.0)
            - test_size: Proportion of data for test split (default 0.2)
        
        Returns:
            Complete EDA report with missing values, outliers, correlations, and visualizations
        """
        report = {
            "dataset_info": {
                "rows": df.shape[0],
                "columns": df.shape[1],
                "column_types": df.dtypes.astype(str).to_dict(),
            },
            "missing_values": MissingValueAnalyzer.analyze(df),
            "column_types": {
                "numeric_columns": df.select_dtypes(include=['number']).columns.tolist(),
                "categorical_columns": df.select_dtypes(include=['object']).columns.tolist(),
                "datetime_columns": df.select_dtypes(include=['datetime']).columns.tolist()
            }
        }
        
        # Add outlier detection
        if include_outliers:
            report["outliers"] = OutlierDetector.analyze(df, zscore_threshold=zscore_threshold)
        
        # Add visualizations and advanced statistics
        if include_visualizations:
            report["correlation_analysis"] = PlotGenerator.generate_correlation_matrix(df)
            report["distribution_analysis"] = PlotGenerator.generate_distribution_plots(df)
            report["categorical_analysis"] = PlotGenerator.generate_categorical_plots(df)
            report["train_test_split"] = PlotGenerator.generate_train_test_split_summary(df, test_size=test_size)
        
        return report
