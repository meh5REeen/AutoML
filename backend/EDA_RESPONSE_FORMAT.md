# EDA API Response Format

## Response Structure

```json
{
  "status": "success",
  "filename": "data.csv",
  "data": {
    "dataset_info": {
      "rows": 1000,
      "columns": 10,
      "column_types": {
        "age": "int64",
        "salary": "float64",
        "department": "object",
        "years_experience": "int64"
      }
    },
    "missing_values": {
      "per_feature": {
        "age": {
          "count": 5,
          "percent": 0.5,
          "non_missing": 995
        },
        "salary": {
          "count": 0,
          "percent": 0.0,
          "non_missing": 1000
        }
      },
      "global_percent": 0.12,
      "total_missing": 12,
      "total_cells": 10000,
      "total_rows": 1000,
      "total_columns": 10
    },
    "column_types": {
      "numeric_columns": ["age", "salary", "years_experience"],
      "categorical_columns": ["department"],
      "datetime_columns": []
    },
    "outliers": {
      "age": {
        "iqr": {
          "method": "IQR",
          "outlier_count": 8,
          "outlier_percent": 0.81,
          "lower_bound": 25.5,
          "upper_bound": 72.5,
          "outlier_indices": [45, 132, 234, ...],
          "outlier_values": [15.2, 18.3, 19.5, ...]
        },
        "zscore": {
          "method": "Z-Score",
          "threshold": 3.0,
          "outlier_count": 3,
          "outlier_percent": 0.3,
          "outlier_indices": [120, 450, 876],
          "outlier_values": [2.5, 95.3, 98.1]
        }
      },
      "salary": {
        "iqr": {...},
        "zscore": {...}
      }
    },
    "correlation_analysis": {
      "correlation_matrix": [
        [1.0, 0.45, 0.32],
        [0.45, 1.0, -0.12],
        [0.32, -0.12, 1.0]
      ],
      "columns": ["age", "salary", "years_experience"],
      "message": null
    },
    "distribution_analysis": {
      "distributions": {
        "age": {
          "values": [5, 12, 28, 45, 67, 89, 102, 95, 82, 65, ...],
          "bins": [18.2, 19.5, 20.8, 22.1, 23.4, 24.7, 26.0, ...],
          "bin_centers": [18.85, 20.15, 21.45, 22.75, ...]
        },
        "salary": {
          "values": [3, 8, 15, 32, 48, 65, 78, 92, 105, 98, ...],
          "bins": [20000, 22000, 24000, 26000, ...],
          "bin_centers": [21000, 23000, 25000, ...]
        },
        "years_experience": {...}
      },
      "statistics": {
        "age": {
          "mean": 42.5,
          "median": 41.0,
          "std": 12.3,
          "min": 18.0,
          "max": 95.0,
          "q1": 32.0,
          "q3": 53.0,
          "skewness": 0.34,
          "kurtosis": -0.12,
          "count": 995
        },
        "salary": {
          "mean": 75000.5,
          "median": 70000.0,
          "std": 25000.3,
          "min": 20000.0,
          "max": 250000.0,
          "q1": 55000.0,
          "q3": 95000.0,
          "skewness": 1.23,
          "kurtosis": 2.45,
          "count": 1000
        },
        "years_experience": {...}
      },
      "columns": ["age", "salary", "years_experience"],
      "message": null
    },
    "categorical_analysis": {
      "value_counts": {
        "department": {
          "labels": ["Sales", "Engineering", "HR", "Finance", "Marketing"],
          "values": [320, 280, 150, 140, 110],
          "unique_count": 5,
          "total_count": 1000
        }
      },
      "columns": ["department"],
      "message": null
    },
    "train_test_split": {
      "total_samples": 1000,
      "train_samples": 800,
      "test_samples": 200,
      "train_percentage": 80.0,
      "test_percentage": 20.0,
      "test_size": 0.2,
      "random_state": 42
    }
  }
}
```

## Usage on Frontend

### Correlation Matrix (Plotly)
```jsx
import Plot from 'react-plotly.js';

const CorrelationPlot = ({ data }) => {
  const corrData = data.correlation_analysis;
  
  return (
    <Plot
      data={[{
        z: corrData.correlation_matrix,
        x: corrData.columns,
        y: corrData.columns,
        type: 'heatmap',
        colorscale: 'RdBu',
        zmid: 0,
        zmin: -1,
        zmax: 1
      }]}
      layout={{ title: 'Correlation Matrix' }}
    />
  );
};
```

### Distribution Plots (Plotly)
```jsx
const DistributionPlots = ({ data }) => {
  const distributions = data.distribution_analysis.distributions;
  const stats = data.distribution_analysis.statistics;
  
  return Object.keys(distributions).map(column => (
    <Plot key={column}
      data={[{
        x: distributions[column].bin_centers,
        y: distributions[column].values,
        type: 'bar',
        name: column
      }]}
      layout={{ title: `Distribution: ${column}` }}
    />
  ));
};
```

### Categorical Plots (Chart.js or Plotly)
```jsx
const CategoricalPlots = ({ data }) => {
  const categories = data.categorical_analysis.value_counts;
  
  return Object.keys(categories).map(column => (
    <Plot key={column}
      data={[{
        x: categories[column].labels,
        y: categories[column].values,
        type: 'bar'
      }]}
      layout={{ title: `Distribution: ${column}` }}
    />
  ));
};
```

### Train/Test Split (Chart.js)
```jsx
import { PieChart, Pie, Cell } from 'recharts';

const TrainTestSplit = ({ data }) => {
  const split = data.train_test_split;
  
  const chartData = [
    { name: 'Train', value: split.train_samples },
    { name: 'Test', value: split.test_samples }
  ];
  
  return (
    <PieChart width={400} height={400}>
      <Pie data={chartData} dataKey="value" label>
        <Cell fill="#2ecc71" />
        <Cell fill="#e74c3c" />
      </Pie>
    </PieChart>
  );
};
```

## Benefits of This Approach

✅ No heavy image rendering on backend
✅ Smaller payload (raw data is 30-40% smaller than base64 images)
✅ Frontend can update visualizations dynamically without re-uploading
✅ Interactive plots (zoom, hover, filters)
✅ Better performance overall
✅ Consistent with modern frontend frameworks
