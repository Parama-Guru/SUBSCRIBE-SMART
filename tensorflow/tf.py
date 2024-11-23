import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau
from scikeras.wrappers import KerasClassifier
import os

# Load the dataset
data = pd.read_csv("/Users/guru/vsc/ml/dataset/ml project.csv")

# Drop irrelevant 'customerID' column
data = data.drop(columns=['customerID'])

data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce')

data['TotalCharges'].fillna(data['TotalCharges'].median(), inplace=True)

X = data.drop(columns=['Churn'])
y = data['Churn']

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

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

X_processed = preprocessor.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42)

def create_model(units=64, learning_rate=0.001, dropout=0.2, num_layers=2):
    print(f"Building model with units={units}, learning_rate={learning_rate}, dropout={dropout}, num_layers={num_layers}")
    model = Sequential()
    model.add(Input(shape=(X_train.shape[1],)))  #Input layer
    
    # hidden layer
    model.add(Dense(units, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(dropout))
    
    for i in range(num_layers - 1):
        model.add(Dense(units // (2 ** (i + 1)), activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(dropout))
    
    # Output layer
    model.add(Dense(1, activation='sigmoid'))
    
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()
    return model

model = KerasClassifier(
    model=create_model, 
    epochs=10, 
    batch_size=32, 
    verbose=0
)

# Adjusted parameter distribution for RandomizedSearchCV
param_dist = {
    'model__units': [64, 128, 256, 512],
    'model__learning_rate': [0.01, 0.005, 0.001, 0.0005],
    'model__dropout': [0.1, 0.2, 0.3],
    'model__num_layers': [2, 3, 4, 5],
    'batch_size': [16, 32, 64, 128, 256],
    'epochs': [10, 20, 30, 40, 50]
}

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, min_lr=0.00001, verbose=1)

n_iterations = 1000 // 5   
random_search = RandomizedSearchCV(estimator=model, param_distributions=param_dist, 
                                   n_iter=n_iterations, cv=3, scoring='accuracy', verbose=1, random_state=42)

print("Starting RandomizedSearchCV...")
random_search_result = random_search.fit(X_train, y_train, callbacks=[reduce_lr])

performance_file = "performance_measure.txt"
with open(performance_file, "w") as f:
    for i in range(len(random_search_result.cv_results_['params'])):
        params = random_search_result.cv_results_['params'][i]
        
        y_pred = random_search_result.best_estimator_.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        f.write(f"Params: {params}\n")
        f.write(f"Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1 Score: {f1}\n")
        f.write("-" * 40 + "\n")

# Save the best model's performance in a separate file
best_performance_file = "best_model_pm.txt"
best_model = random_search_result.best_estimator_
y_pred_best = best_model.predict(X_test)

# Calculate metrics for the best model
best_accuracy = accuracy_score(y_test, y_pred_best)
best_precision = precision_score(y_test, y_pred_best)
best_recall = recall_score(y_test, y_pred_best)
best_f1 = f1_score(y_test, y_pred_best)


with open(best_performance_file, "w") as f:
    f.write(f"Best Params: {random_search_result.best_params_}\n")
    f.write(f"Accuracy: {best_accuracy}\n")
    f.write(f"Precision: {best_precision}\n")
    f.write(f"Recall: {best_recall}\n")
    f.write(f"F1 Score: {best_f1}\n")

os.makedirs('best model', exist_ok=True)
best_model.model_.save('best model/deep_model.keras')
print("Best model saved as 'best model/deep_model/deep_model.keras'")