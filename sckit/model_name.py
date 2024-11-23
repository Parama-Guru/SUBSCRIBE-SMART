import joblib

# Load the model
model = joblib.load("/Users/guru/vsc/ml/myself/sckit/best_model.joblib")

# Print the model's class to identify its type
print(type(model))