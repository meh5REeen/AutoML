import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import base64
from typing import Dict, Any, List


class PlotGenerator:
    """Generate visualizations for EDA analysis."""
    
    @staticmethod
    def _fig_to_base64(fig) -> str:
        """Convert matplotlib figure to base64 string."""
        buffer = BytesIO()
        fig.savefig(buffer, format='png', bbox_inches='tight', dpi=100)
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.read()).decode()
        plt.close(fig)
        return image_base64
    
    @staticmethod
    def generate_correlation_matrix(df: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate correlation matrix visualization.
        
        Returns:
            Dictionary with:
            - correlation_matrix: Correlation values as dict
            - plot: Base64 encoded PNG image
        """
        numeric_df = df.select_dtypes(include=['number'])
        
        if numeric_df.empty or len(numeric_df.columns) < 2:
            return {
                "correlation_matrix": {},
                "plot": None,
                "message": "Not enough numeric columns for correlation analysis"
            }
        
        # Calculate correlation matrix
        corr_matrix = numeric_df.corr()
        
        # Create visualization
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                    center=0, square=True, ax=ax, cbar_kws={"shrink": 0.8})
        ax.set_title('Correlation Matrix', fontsize=16, fontweight='bold', pad=20)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        
        plot_base64 = PlotGenerator._fig_to_base64(fig)
        
        return {
            "correlation_matrix": corr_matrix.to_dict(),
            "plot": plot_base64,
            "columns": numeric_df.columns.tolist()
        }
    
    @staticmethod
    def generate_distribution_plots(df: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate distribution plots for numerical features.
        
        Returns:
            Dictionary with:
            - plots: Dictionary mapping column names to base64 encoded images
            - statistics: Basic statistics for each numeric column
        """
        numeric_columns = df.select_dtypes(include=['number']).columns.tolist()
        
        if not numeric_columns:
            return {
                "plots": {},
                "statistics": {},
                "message": "No numeric columns found"
            }
        
        plots = {}
        statistics = {}
        
        # Create subplots for all numeric columns
        n_cols = min(3, len(numeric_columns))
        n_rows = (len(numeric_columns) + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
        
        # Flatten axes array for easier iteration
        if isinstance(axes, np.ndarray):
            axes = axes.flatten()
        else:
            axes = [axes]
        
        for idx, column in enumerate(numeric_columns):
            ax = axes[idx]
            data = df[column].dropna()
            
            # Create histogram with KDE
            ax.hist(data, bins=30, edgecolor='black', alpha=0.7, color='skyblue')
            ax.set_title(f'Distribution: {column}', fontweight='bold')
            ax.set_xlabel('Value')
            ax.set_ylabel('Frequency')
            ax.grid(axis='y', alpha=0.3)
            
            # Calculate statistics
            statistics[column] = {
                "mean": float(data.mean()),
                "median": float(data.median()),
                "std": float(data.std()),
                "min": float(data.min()),
                "max": float(data.max()),
                "skewness": float(data.skew()),
                "kurtosis": float(data.kurtosis()),
                "q1": float(data.quantile(0.25)),
                "q3": float(data.quantile(0.75))
            }
        
        # Hide unused subplots
        for idx in range(len(numeric_columns), len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        plot_base64 = PlotGenerator._fig_to_base64(fig)
        
        return {
            "plot": plot_base64,
            "statistics": statistics,
            "columns": numeric_columns
        }
    
    @staticmethod
    def generate_categorical_plots(df: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate bar plots for categorical features.
        
        Returns:
            Dictionary with:
            - plots: Dictionary mapping column names to base64 encoded images
            - value_counts: Value distribution for each categorical column
        """
        categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
        
        if not categorical_columns:
            return {
                "plots": {},
                "value_counts": {},
                "message": "No categorical columns found"
            }
        
        value_counts = {}
        
        # Create subplots for all categorical columns
        n_cols = min(3, len(categorical_columns))
        n_rows = (len(categorical_columns) + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
        
        # Flatten axes array for easier iteration
        if isinstance(axes, np.ndarray):
            axes = axes.flatten()
        else:
            axes = [axes]
        
        for idx, column in enumerate(categorical_columns):
            ax = axes[idx]
            
            # Get value counts
            vc = df[column].value_counts().head(10)  # Limit to top 10
            value_counts[column] = {
                "unique_count": int(df[column].nunique()),
                "top_values": vc.to_dict()
            }
            
            # Create bar plot
            vc.plot(kind='bar', ax=ax, color='steelblue', edgecolor='black')
            ax.set_title(f'Value Distribution: {column}', fontweight='bold')
            ax.set_xlabel('Category')
            ax.set_ylabel('Count')
            ax.tick_params(axis='x', rotation=45)
            ax.grid(axis='y', alpha=0.3)
        
        # Hide unused subplots
        for idx in range(len(categorical_columns), len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        plot_base64 = PlotGenerator._fig_to_base64(fig)
        
        return {
            "plot": plot_base64,
            "value_counts": value_counts,
            "columns": categorical_columns
        }
    
    @staticmethod
    def generate_train_test_split_summary(df: pd.DataFrame, test_size: float = 0.2, 
                                         random_state: int = 42) -> Dict[str, Any]:
        """
        Generate train/test split summary with visualization.
        
        Parameters:
            - test_size: Proportion of dataset to use for testing (default 0.2)
            - random_state: Random seed for reproducibility
        
        Returns:
            Dictionary with:
            - split_summary: Statistics about the split
            - plot: Base64 encoded visualization
        """
        from sklearn.model_selection import train_test_split
        
        # Create dummy split for visualization purposes
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
            "random_state": random_state
        }
        
        # Create visualization
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Pie chart
        labels = ['Train', 'Test']
        sizes = [train_rows, test_rows]
        colors = ['#2ecc71', '#e74c3c']
        explode = (0.05, 0)
        
        axes[0].pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
                   shadow=True, startangle=90, textprops={'fontsize': 12, 'fontweight': 'bold'})
        axes[0].set_title('Train/Test Split Distribution', fontweight='bold', fontsize=14)
        
        # Bar chart
        axes[1].bar(labels, sizes, color=colors, edgecolor='black', alpha=0.8)
        axes[1].set_ylabel('Number of Samples', fontweight='bold')
        axes[1].set_title('Train/Test Split Count', fontweight='bold', fontsize=14)
        axes[1].grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for i, v in enumerate(sizes):
            axes[1].text(i, v + max(sizes)*0.01, str(v), ha='center', 
                        fontweight='bold', fontsize=12)
        
        plt.tight_layout()
        plot_base64 = PlotGenerator._fig_to_base64(fig)
        
        return {
            "split_summary": split_data,
            "plot": plot_base64
        }
