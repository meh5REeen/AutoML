import pandas as pd
from sklearn.preprocessing import OrdinalEncoder, StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split

def handle_missing_values(df,strategy="None",fill_value=None):
    for col in df.columns:
        if df[col].isnull().sum() > 0 :
            if df[col].dtype in ['int64','float64']:
                if strategy == "Mean":
                    df[col].fillna(df[col].mean(),inplace=True)
                elif strategy == "Median":
                    df[col].fillna(df[col].median(),inplace=True)
                elif strategy == "Mode":
                    df[col].fillna(df[col].mode()[0],inplace=True)
                elif strategy == "Constant" and fill_value is not None:
                    df[col].fillna(fill_value,inplace=True)
            else:
                if strategy =="Mode":
                    df[col].fillna(df[col].mode()[0],inplace=True)
                elif strategy == "Constant" and fill_value is not None:
                    df[col].fillna(fill_value,inplace=True)
    return df

def encode_categorical_variables(df,encoding_type="OneHot"):
    categorical_cols = df.select_dtypes(include=['object','category']).columns
    categorical_mappings = {}
    
    if encoding_type == "OneHot":
        # Store original values before encoding
        for col in categorical_cols:
            categorical_mappings[col] = {
                "type": "onehot",
                "original_values": df[col].unique().tolist()
            }
        df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    elif encoding_type == "Ordinal":
        encoder = OrdinalEncoder()
        # Store the mapping for each categorical column
        for col in categorical_cols:
            unique_vals = df[col].unique()
            categorical_mappings[col] = {
                "type": "ordinal",
                "mapping": {str(val): idx for idx, val in enumerate(unique_vals)}
            }
        df[categorical_cols]=encoder.fit_transform(df[categorical_cols])
    
    return df, categorical_mappings


def scale_numerical_features(df,scaling_type="Standard"):
    numerical_cols = df.select_dtypes(include=['int64','float64']).columns
    if scaling_type == "Standard":
        scaler = StandardScaler()
    elif scaling_type == "MinMax":
        scaler = MinMaxScaler()
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
    return df

def handle_outliers(df,method="Remove",threshold=3):
    numerical_cols = df.select_dtypes(include=['int64','float64']).columns
    if method == "Remove":
        for col in numerical_cols:
            q1 = df[col].quantile(0.25)
            q99 = df[col].quantile(0.99)
            df = df[(df[col] >= q1 - threshold * (q99 - q1)) & (df[col] <= q99 + threshold * (q99 - q1))]
    elif method == "Cap":
        for col in numerical_cols:
            q1 = df[col].quantile(0.01)
            q99 = df[col].quantile(0.99)
            df[col] = df[col].clip(lower=q1,upper=q99)
    return df


def splitting_data(df,target,test_size=0.2):
    X = df.drop(columns=[target])
    y=df[target]
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=test_size,random_state=42)
    return X_train,X_test,y_train,y_test

def remove_duplicates(df):
    return df.drop_duplicates()
