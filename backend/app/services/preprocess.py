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
    if encoding_type == "OneHot":
        df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    elif encoding_type == "Ordinal":
        df[categorical_cols]=OrdinalEncoder().fit_transform(df[categorical_cols])
    return df

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
    return X_train[:5],X_test[:5],y_train[:5],y_test[:5]


