import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler


def preprocess_csv(input_file, output_file, numeric_columns):
    # Read the CSV file
    df = pd.read_csv(input_file)
    
    # Replace non-numeric values with a default value (e.g., 0) in specified columns
    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        df[col] = df[col].replace(np.nan, 0)  # Replace NaN values with 0
    
    # Save the cleaned data to a new CSV file
    df.to_csv(output_file, index=False)


def load_and_predict(input_file, model_file, numeric_columns, output_image):
    # Preprocess the CSV file
    output_file = 'preprocessed.csv'
    preprocess_csv(input_file, output_file, numeric_columns)
    
    # Load the pre-trained model
    model = joblib.load(model_file)
    
    # Load the preprocessed data
    data = pd.read_csv(output_file)
    
    # Prepare features (X) and target (y)
    X = data.drop(columns=[])
    y = data['Comments']
    
    # Scale the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Predict using the SVM model
    y_pred = model.predict(X_scaled)
    
    # Generate a graph
    plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=y_pred, cmap=plt.cm.Paired)
    plt.savefig(output_image)
    plt.close()
    
    return output_image


input_file = 'C:\\Users\\shail\\Dropbox\\My PC (LAPTOP-674MEPPR)\\Desktop\\Final Year\\Code\\-DQP4fpmJpc\\comments.csv'
model_path = 'model.joblib'
numeric_columns = []
output_image = 'C:\\Users\\shail\\Dropbox\\My PC (LAPTOP-674MEPPR)\\Desktop\\Final Year\\Code\\static'

load_and_predict(input_file, model_path, numeric_columns, output_image)