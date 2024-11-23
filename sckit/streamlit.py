import pandas as pd
import joblib
import streamlit as st
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


model = joblib.load("/Users/guru/vsc/ml/myself/sckit/best_model.joblib")
# model = load_model('best model/deep_model.keras')
dataset = pd.read_csv("/Users/guru/vsc/ml/myself/dataset/ml project.csv")
if 'customerID' in dataset.columns:
        dataset.drop(columns=['customerID'], inplace=True)

def preprocess_with_user_input(input_data):

    data = pd.concat([input_data, dataset], ignore_index=True)

    data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce')

    data['TotalCharges'].fillna(data['TotalCharges'].median(), inplace=True)

    X = data.drop(columns=['Churn'])

    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

    numerical_pipeline = Pipeline(steps=[
    ('scaler', StandardScaler())
])

    categorical_pipeline = Pipeline(steps=[
    ('onehot', OneHotEncoder(drop='first'))
])

    preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_pipeline, numerical_cols),
        ('cat', categorical_pipeline, categorical_cols)
    ])

    preprocessed_data = preprocessor.fit_transform(X)
    
    return preprocessed_data[0].reshape(1, -1)  


st.title("Customer Churn Prediction(Sckit learn)")


st.header("Enter Customer Data")
user_input = {}
for col in dataset.columns:
    if col == 'TotalCharges':

        if 'TotalCharges' in dataset.columns:
            dataset['TotalCharges'] = pd.to_numeric(dataset['TotalCharges'], errors='coerce')
            dataset['TotalCharges'].fillna(dataset['TotalCharges'].median(), inplace=True)
        user_input[col] = st.number_input(f"Enter {col}", min_value=float(dataset[col].min()), max_value=float(dataset[col].max()))
    elif col != 'customerID' and col != 'Churn':
        if dataset[col].dtype == 'object':
            user_input[col] = st.selectbox(f"Select {col}", options=dataset[col].unique())
        else:
            user_input[col] = st.number_input(f"Enter {col}", min_value=float(dataset[col].min()), max_value=float(dataset[col].max()))


input_df = pd.DataFrame([user_input])

if st.button("Predict Churn"):
    processed_input = preprocess_with_user_input(input_df)
    try:
        y_pred = model.predict(processed_input)
        prediction = (y_pred > 0.5).astype("int32") 
        st.write("Prediction: ", "No Churn" if prediction[0] == 1 else "Churn")
    except ValueError as e:
        st.error(f"Prediction error: {e}")