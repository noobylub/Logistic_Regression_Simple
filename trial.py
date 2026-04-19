import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

# Import the logistic regression model from LR.py
from LR import LogisticRegressionModel

# Check for CUDA availability, if none just use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load the data 
try:
    data = pd.read_csv('Compiled_Reviews.txt', sep='\t')
    print("Data loaded successfully. Columns:", data.columns.tolist())
except Exception as e:
    print(f"Error loading data: {e}")
    exit()

texts = data['REVIEW'].fillna('').values  
label_mapping = {'positive': 1, 'negative': 0}
labels = data['RATING'].map(label_mapping).astype(int).values

# Vectorise. the dataset, in this case I am sticking to TF-IDF, but you can use other methods like word embeddings or BERT embeddings for better performance.
vectorizer = TfidfVectorizer(max_features=5000)  # Limit to top 5000 features for simplicity
X = vectorizer.fit_transform(texts).toarray()
Y = labels
print("Data preprocessing with TF-IDF")
# Seeing how the data looks like 
for i in range(5):
    print(f"Review: {texts[i][:100]}... | Label: {labels[i]}")
    print(X[i][:50])  # Print all features for the review
    print(Y[i])



vectorizer_count = CountVectorizer(max_features=5000)
X_count = vectorizer_count.fit_transform(texts).toarray()
print("-----"*100)
print("Data preprocessing with CountVectorizer")
for i in range(5):
    print(f"Review: {texts[i][:100]}... | Label: {labels[i]}")
    print(X_count[i][:50])  # Print all features for the review
    print(Y[i])
