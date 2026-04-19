import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# Import the logistic regression model from LR.py
from LR import LogisticRegressionModel
from data_load import load_data


print("CountVectorizer Model")
print("-----"*100)

# Check for CUDA availability, if none just use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

np.random.seed(42)  # For reproducibility
X_train, y_train, X_test, y_test = load_data(x_data='REVIEW', y_data='RATING', label_mapping={'positive': 1, 'negative': 0}, device=device, vectorizer=CountVectorizer, max_features=5000)
# Building the logistic regression model
# We will try to modify the number of layers to see if they have an effect

# To ensure efficiency on GPU we use TensorLoader and DataLoader to handle batching
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
    print(f"Epoch {epoch+1}/{num_epochs}, Avg Loss: {avg_loss:.4f}")

# Saving the model
torch.save(model.state_dict(), 'Count_Model.pt')

# Validation (optional, to tune)
model.eval()
with torch.no_grad():
    val_outputs = model(X_test).cpu().numpy()
    val_auc = roc_auc_score(y_test.cpu().numpy(), val_outputs)
    print(f"Validation AUC: {val_auc:.4f}")

# Evaluation on test set
with torch.no_grad():
    test_outputs = model(X_test).cpu().numpy()
    test_labels = y_test.cpu().numpy()
    auc = roc_auc_score(test_labels, test_outputs)
    print(f"Test AUC: {auc:.4f}")
    
    # Calculate accuracy
    predictions = (test_outputs > 0.5).astype(int)
    accuracy = np.mean(predictions == test_labels)
    print(f"Test Accuracy: {accuracy:.4f}")
