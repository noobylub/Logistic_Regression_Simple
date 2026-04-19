import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

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
label_mapping ={'positive': 1, 'negative': 0}
labels = data['RATING'].map(label_mapping).astype(int).values

# Vectorise. the dataset, in this case I am sticking to TF-IDF, but you can use other methods like word embeddings or BERT embeddings for better performance.
vectorizer = TfidfVectorizer(max_features=5000)  # Limit to top 5000 features for simplicity
X = vectorizer.fit_transform(texts).toarray()
Y = labels

# Split data into training and testing sets
# For this we just use 20 percent for testing, 80 percent for training
np.random.seed(42)  # For reproducibility
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)


X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
y_train = torch.tensor(y_train, dtype=torch.float32).to(device)
X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
y_test = torch.tensor(y_test, dtype=torch.float32).to(device)

# Building the logistic regression model
# We will try to modify the number of layers to see if they have an effect
class LogisticRegressionModel(nn.Module):
    def __init__(self, input_dim):
        super(LogisticRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, 1)  

    def forward(self, x):
        return torch.sigmoid(self.linear(x)).squeeze(1)  
    
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
