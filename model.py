import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import auc, roc_auc_score, roc_curve
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

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


# Vectorise with TF-IDF in this run
# Limit to top 5000 features for simplicity and so that it can reasonably run
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(texts).toarray()
Y = labels

# Split data into training and testing sets
# 30 percent for testing 
np.random.seed(42)  
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3)


X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
y_train = torch.tensor(y_train, dtype=torch.float32).to(device)
X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
y_test = torch.tensor(y_test, dtype=torch.float32).to(device)

# Building the logistic regression model
# To ensure efficiency on GPU, TensorLoader and DataLoader to handle batching
train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

input_dim = X_train.shape[1]
model = LogisticRegressionModel(input_dim).to(device)

criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
    # Iterate through the data in batches, so it is faster and more efficient
    for inputs, targets in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    avg_loss = epoch_loss / len(train_loader)
    # Reporting for every epoch 
    print(f"Epoch {epoch+1}/{num_epochs}, Avg Loss: {avg_loss:.2f}")


# Evaluation on test set
with torch.no_grad():
    test_outputs = model(X_test).cpu().numpy()
    test_labels = y_test.cpu().numpy()

    predictions = (test_outputs > 0.5).astype(int)
    accuracy = np.mean(predictions == test_labels)
    print(f"Test Accuracy: {accuracy:.4f}")
    fpr, tpr, thresholds = roc_curve(test_labels, test_outputs)
    auc_score = auc(fpr, tpr)
    print(f"Test AUC: {auc_score:.4f}")
    

