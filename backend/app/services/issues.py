import pandas as pd
import numpy as np
from datetime import datetime


class IssueDetector:
    """Detect and categorize data quality issues in datasets"""
    
    @staticmethod
    def detect_all_issues(df_original, df_processed, preprocessing_params):
        """
        Detect all data quality issues between original and processed datasets
        
        Args:
            df_original: Original dataset before preprocessing
            df_processed: Dataset after preprocessing
            preprocessing_params: Dict with preprocessing parameters used
            
        Returns:
            Dict with all detected issues and recommendations
        """
        issues = []
        
        # 1. Missing values
        missing_issues = IssueDetector._detect_missing_values(df_original)
        issues.extend(missing_issues)
        
        # 2. Duplicates
        duplicate_issues = IssueDetector._detect_duplicates(df_original)
        issues.extend(duplicate_issues)
        
        # 3. Outliers
        outlier_issues = IssueDetector._detect_outliers(df_original)
        issues.extend(outlier_issues)
        
        # 4. Data type inconsistencies
        type_issues = IssueDetector._detect_type_inconsistencies(df_original)
        issues.extend(type_issues)
        
        # 5. Class imbalance (for target variable if classification)
        imbalance_issues = IssueDetector._detect_class_imbalance(df_original)
        issues.extend(imbalance_issues)
        
        # 6. High cardinality categorical features
        cardinality_issues = IssueDetector._detect_high_cardinality(df_original)
        issues.extend(cardinality_issues)
        
        return {
            "timestamp": datetime.now().isoformat(),
            "issues_count": len(issues),
            "issues_detected": issues,
            "severity_summary": IssueDetector._summarize_severity(issues),
            "data_shape_before": df_original.shape,
            "data_shape_after": df_processed.shape,
            "rows_removed": df_original.shape[0] - df_processed.shape[0],
            "recommendations": IssueDetector._generate_recommendations(issues)
        }
    
    @staticmethod
    def _detect_missing_values(df):
        """Detect missing values in dataset"""
        issues = []
        for col in df.columns:
            missing_count = df[col].isnull().sum()
            if missing_count > 0:
                missing_percent = (missing_count / len(df)) * 100
                severity = "high" if missing_percent > 30 else "medium" if missing_percent > 10 else "low"
                issues.append({
                    "type": "missing_values",
                    "column": col,
                    "count": int(missing_count),
                    "percent": round(missing_percent, 2),
                    "severity": severity
                })
        return issues
    
    @staticmethod
    def _detect_duplicates(df):
        """Detect duplicate rows"""
        issues = []
        duplicate_count = df.duplicated().sum()
        if duplicate_count > 0:
            duplicate_percent = (duplicate_count / len(df)) * 100
            severity = "high" if duplicate_percent > 5 else "medium" if duplicate_percent > 1 else "low"
            issues.append({
                "type": "duplicate_rows",
                "column": "Dataset",
                "count": int(duplicate_count),
                "percent": round(duplicate_percent, 2),
                "severity": severity
            })
        return issues
    
    @staticmethod
    def _detect_outliers(df):
        """Detect outliers in numerical columns"""
        issues = []
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numerical_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outlier_count = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
            if outlier_count > 0:
                outlier_percent = (outlier_count / len(df)) * 100
                severity = "high" if outlier_percent > 5 else "medium" if outlier_percent > 1 else "low"
                issues.append({
                    "type": "outliers",
                    "column": col,
                    "method": "IQR",
                    "count": int(outlier_count),
                    "percent": round(outlier_percent, 2),
                    "bounds": {"lower": round(float(lower_bound), 2), "upper": round(float(upper_bound), 2)},
                    "severity": severity
                })
        return issues
    
    @staticmethod
    def _detect_type_inconsistencies(df):
        """Detect columns with potential type issues"""
        issues = []
        for col in df.columns:
            if df[col].dtype == 'object':
                # Check if object column contains mixed types
                try:
                    pd.to_numeric(df[col], errors='coerce')
                    non_numeric = df[col].apply(lambda x: pd.isna(x) or not str(x).replace('.', '', 1).replace('-', '', 1).isdigit()).sum()
                    if non_numeric > 0 and non_numeric < len(df):
                        issues.append({
                            "type": "type_inconsistency",
                            "column": col,
                            "description": "Column contains both numeric and non-numeric values",
                            "severity": "medium"
                        })
                except:
                    pass
        return issues
    
    @staticmethod
    def _detect_class_imbalance(df):
        """Detect class imbalance in categorical columns"""
        issues = []
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        
        for col in categorical_cols:
            if df[col].nunique() <= 10:  # Likely a target or class column
                value_counts = df[col].value_counts(normalize=True)
                min_ratio = value_counts.min()
                max_ratio = value_counts.max()
                imbalance_ratio = max_ratio / min_ratio if min_ratio > 0 else 0
                
                if imbalance_ratio > 3:  # Significant imbalance
                    severity = "high" if imbalance_ratio > 10 else "medium"
                    issues.append({
                        "type": "class_imbalance",
                        "column": col,
                        "imbalance_ratio": round(imbalance_ratio, 2),
                        "class_distribution": value_counts.to_dict(),
                        "severity": severity
                    })
        return issues
    
    @staticmethod
    def _detect_high_cardinality(df):
        """Detect high cardinality categorical features"""
        issues = []
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        
        for col in categorical_cols:
            cardinality = df[col].nunique()
            cardinality_ratio = cardinality / len(df)
            
            if cardinality > 50 or cardinality_ratio > 0.5:
                severity = "high" if cardinality > 100 else "medium"
                issues.append({
                    "type": "high_cardinality",
                    "column": col,
                    "unique_values": int(cardinality),
                    "cardinality_ratio": round(cardinality_ratio, 2),
                    "severity": severity
                })
        return issues
    
    @staticmethod
    def _summarize_severity(issues):
        """Summarize issues by severity level"""
        severity_count = {"high": 0, "medium": 0, "low": 0}
        for issue in issues:
            severity = issue.get("severity", "low")
            severity_count[severity] = severity_count.get(severity, 0) + 1
        return severity_count
    
    @staticmethod
    def _generate_recommendations(issues):
        """Generate recommendations based on detected issues"""
        recommendations = []
        issue_types = [issue["type"] for issue in issues]
        
        if "missing_values" in issue_types:
            recommendations.append("Handle missing values using mean/median/mode imputation or remove rows")
        
        if "duplicate_rows" in issue_types:
            recommendations.append("Remove duplicate rows from dataset")
        
        if "outliers" in issue_types:
            recommendations.append("Consider removing or capping outliers using IQR method")
        
        if "class_imbalance" in issue_types:
            recommendations.append("Use resampling techniques (SMOTE) or class weights for balanced training")
        
        if "high_cardinality" in issue_types:
            recommendations.append("Apply feature selection or dimensionality reduction for high cardinality features")
        
        return recommendations
