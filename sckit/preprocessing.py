import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

def preprocess_data(file_path):
    data = pd.read_csv(file_path)
    
    if 'customerID' in data.columns:
        data.drop(columns=['customerID'], inplace=True)

    if 'TotalCharges' in data.columns:
        data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce')
        data['TotalCharges'].fillna(data['TotalCharges'].median(), inplace=True)

    X = data.drop(columns=['Churn'])
    y = LabelEncoder().fit_transform(data['Churn'])

    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    numerical_cols = X.select_dtypes(include=['float64', 'int64']).columns.tolist()

    label_encoders = {}
    for col in categorical_cols:
        if X[col].nunique() == 2:  
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col])
            label_encoders[col] = le  

    categorical_cols = [col for col in categorical_cols if X[col].nunique() > 2]

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(drop='first', sparse_output=False))
    ])
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ],
        remainder='passthrough'
    )

    X_processed = preprocessor.fit_transform(X)
    X_train, X_temp, y_train, y_temp = train_test_split(X_processed, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    os.makedirs("preprocessed_dataset", exist_ok=True)

    pd.DataFrame(X_train).to_csv('preprocessed_dataset/X_train.csv', index=False)
    pd.DataFrame(X_val).to_csv('preprocessed_dataset/X_val.csv', index=False)
    pd.DataFrame(X_test).to_csv('preprocessed_dataset/X_test.csv', index=False)
    pd.DataFrame(y_train, columns=['Churn']).to_csv('preprocessed_dataset/y_train.csv', index=False)
    pd.DataFrame(y_val, columns=['Churn']).to_csv('preprocessed_dataset/y_val.csv', index=False)
    pd.DataFrame(y_test, columns=['Churn']).to_csv('preprocessed_dataset/y_test.csv', index=False)

# Run preprocessing if the script is executed directly
if __name__ == "__main__":
    file_path = "/Users/guru/vsc/ml/myself/dataset/ml project.csv"  
    preprocess_data(file_path)