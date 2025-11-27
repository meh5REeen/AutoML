import pandas as pd
import numpy as np
from typing import Dict, Any, List


class DataGenerator:
    """Generate raw data for frontend visualizations (no image encoding)."""
    
    @staticmethod
    def generate_correlation_matrix(df: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate correlation matrix data.
        
        Returns:
            Dictionary with:
            - correlation_matrix: 2D array of correlation values
            - columns: Column names in order
            - message: Info message if not enough numeric columns
        """
        numeric_df = df.select_dtypes(include=['number'])
        
        if numeric_df.empty or len(numeric_df.columns) < 2:
            return {
                "correlation_matrix": [],
                "columns": [],
                "message": "Not enough numeric columns for correlation analysis"
            }
        
        # Calculate correlation matrix
        corr_matrix = numeric_df.corr()
        
        return {
            "correlation_matrix": corr_matrix.values.tolist(),
            "columns": numeric_df.columns.tolist(),
            "message": None
        }
    
    @staticmethod
    def generate_distribution_data(df: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate distribution data for numerical features.
        
        Returns:
            Dictionary with:
            - distributions: Per-column histogram data
            - statistics: Descriptive statistics for each numeric column
            - columns: List of numeric column names
        """
        numeric_columns = df.select_dtypes(include=['number']).columns.tolist()
        
        if not numeric_columns:
            return {
                "distributions": {},
                "statistics": {},
                "columns": [],
                "message": "No numeric columns found"
            }
        
        distributions = {}
        statistics = {}
        
        for column in numeric_columns:
            data = df[column].dropna()
            
            # Generate histogram data (30 bins)
            hist_values, bin_edges = np.histogram(data, bins=30)
            
            distributions[column] = {
                "values": hist_values.tolist(),
                "bins": bin_edges.tolist(),
                "bin_centers": ((bin_edges[:-1] + bin_edges[1:]) / 2).tolist()
            }
            
            # Calculate statistics
            statistics[column] = {
                "mean": float(data.mean()),
                "median": float(data.median()),
                "std": float(data.std()),
                "min": float(data.min()),
                "max": float(data.max()),
                "q1": float(data.quantile(0.25)),
                "q3": float(data.quantile(0.75)),
                "skewness": float(data.skew()),
                "kurtosis": float(data.kurtosis()),
                "count": int(len(data))
            }
        
        return {
            "distributions": distributions,
            "statistics": statistics,
            "columns": numeric_columns,
            "message": None
        }
    
    @staticmethod
    def generate_categorical_data(df: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate categorical data for categorical features.
        
        Returns:
            Dictionary with:
            - value_counts: Value distribution for each categorical column
            - columns: List of categorical column names
        """
        categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
        
        if not categorical_columns:
            return {
                "value_counts": {},
                "columns": [],
                "message": "No categorical columns found"
            }
        
        value_counts = {}
        
        for column in categorical_columns:
            vc = df[column].value_counts().head(15)  # Limit to top 15
            
            value_counts[column] = {
                "labels": vc.index.tolist(),
                "values": vc.values.tolist(),
                "unique_count": int(df[column].nunique()),
                "total_count": int(len(df[column]))
            }
        
        return {
            "value_counts": value_counts,
            "columns": categorical_columns,
            "message": None
        }
    
    @staticmethod
    def generate_train_test_split_data(df: pd.DataFrame, test_size: float = 0.2) -> Dict[str, Any]:
        """
        Generate train/test split summary data.
        
        Parameters:
            - test_size: Proportion of dataset to use for testing (default 0.2)
        
        Returns:
            Dictionary with:
            - split_summary: Statistics about the split
        """
        total_rows = len(df)
        test_rows = int(total_rows * test_size)
        train_rows = total_rows - test_rows
        
        split_data = {
            "total_samples": total_rows,
            "train_samples": train_rows,
            "test_samples": test_rows,
            "train_percentage": round((train_rows / total_rows) * 100, 2),
            "test_percentage": round((test_rows / total_rows) * 100, 2),
            "test_size": test_size,
            "random_state": 42
        }
        
        return split_data
