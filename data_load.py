
from sklearn.model_selection import train_test_split
import torch
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

def load_data(x_data:str, y_data:str, label_mapping:dict, device, vectorizer=CountVectorizer, max_features=5000, test_size=0.3):
    
    # Load the data 
    try:
        data = pd.read_csv('Compiled_Reviews.txt', sep='\t')
        print("Data loaded successfully. Columns:", data.columns.tolist())
    except Exception as e:
        print(f"Error loading data: {e}")
        exit()

    texts = data[x_data].fillna('').values  
    label_mapping = label_mapping
    labels = data[y_data].map(label_mapping).astype(int).values

    X = vectorizer(max_features=max_features).fit_transform(texts).toarray()
    Y = labels

    # Split data into training and testing sets
    # For this we just use 20 percent for testing, 80 percent for training
    np.random.seed(42)  # For reproducibility
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size)


    X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train = torch.tensor(y_train, dtype=torch.float32).to(device)
    X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_test = torch.tensor(y_test, dtype=torch.float32).to(device)

    return X_train, y_train, X_test, y_test

