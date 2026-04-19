# Load first model 

from pyexpat import model



from sklearn.metrics import precision_recall_fscore_support
import torch
from LR import LogisticRegressionModel
from data_load import load_data
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics import roc_auc_score

# Check for CUDA availability, if none just use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Method for bootstrapping 
def _model_predict(model, X):
    with torch.no_grad():
        outputs = model(torch.tensor(X, dtype=torch.float32).to(device)).cpu().numpy().squeeze()
    return outputs



def _draw_bootstrap_sample(X, y):
    n = len(y)
    indices = np.random.choice(n, n, replace=True)
    return X[indices], y[indices]

def bootstrap_accuracy(data, model, num_samples):
    scores = []
    for i in range(num_samples):
        data_bs = _draw_bootstrap_sample(data[0],data[1])
        pred = _model_predict(model, data_bs[0]) > 0.5
        score=np.mean(pred==data_bs[1])
        scores.append(score)
    scores=np.sort(scores)
    return (np.mean(scores), scores[int(len(scores)*0.025)], scores[int(len(scores)*0.975)])

# TFDIF Vectorizer Model
# Load the dataset 
X_train, y_train, X_test, y_test = load_data(
    x_data='REVIEW', 
    y_data='RATING', 
    label_mapping={'positive': 1, 'negative': 0}, 
    device=device, 
    vectorizer=TfidfVectorizer, 
    max_features=5000)

# Load the model
model_TFIDF = LogisticRegressionModel(input_dim=5000).to(device)  # Ensure input_dim matches your model's expected input size
model_TFIDF.load_state_dict(torch.load('TFIDF_Model.pt'))
model_TFIDF.eval()  

# Bootstraping 
mean_acc, ci_low, ci_high = bootstrap_accuracy((X_test.cpu().numpy(), y_test.cpu().numpy()), model_TFIDF, num_samples=1000)
print(f"TFIDF Model: Mean accuracy={mean_acc:.4f}, 95% CI=({ci_low:.4f}, {ci_high:.4f})")



# COUNT Vectorizer Model
X_train, y_train, X_test, y_test = load_data(
    x_data='REVIEW', 
    y_data='RATING', 
    label_mapping={'positive': 1, 'negative': 0}, 
    device=device, 
    vectorizer=CountVectorizer, 
    max_features=5000)


# Load second model
model_Count = LogisticRegressionModel(input_dim=5000).to(device)  # Ensure input_dim matches your model's expected input size
model_Count.load_state_dict(torch.load('Count_Model.pt'))
model_Count.eval()  

# Bootstraping
mean_acc, ci_low, ci_high = bootstrap_accuracy((X_test.cpu().numpy(), y_test.cpu().numpy()), model_Count, num_samples=1000)
print(f"Count Model: Mean accuracy={mean_acc:.4f}, 95% CI=({ci_low:.4f}, {ci_high:.4f})")






# Calculating P Value 
_, _, X_test_tfidf, y_test_tfidf = load_data(
    x_data='REVIEW',
    y_data='RATING',
    label_mapping={'positive': 1, 'negative': 0},
    device=device,
    vectorizer=TfidfVectorizer,
    max_features=5000
)
_, _, X_test_count, y_test_count = load_data(
    x_data='REVIEW',
    y_data='RATING',
    label_mapping={'positive': 1, 'negative': 0},
    device=device,
    vectorizer=CountVectorizer,
    max_features=5000
)

# Convert to numpy
X_tfidf = X_test_tfidf.cpu().numpy()
y_tfidf = y_test_tfidf.cpu().numpy()
X_count = X_test_count.cpu().numpy()
y_count = y_test_count.cpu().numpy()

def bootstrap_auc_pvalue(X1, y1, model1, X2, y2, model2, num_samples=1000):
    auc_diffs = []
    n = len(y1)
    for _ in range(num_samples):
        indices = np.random.choice(n, n, replace=True)
        X1_bs, y1_bs = X1[indices], y1[indices]
        X2_bs, y2_bs = X2[indices], y2[indices]
        probs1 = _model_predict(model1, X1_bs)
        probs2 = _model_predict(model2, X2_bs)
        auc1 = roc_auc_score(y1_bs, probs1)
        auc2 = roc_auc_score(y2_bs, probs2)
       
        auc_diffs.append(auc2 - auc1)

    auc_diffs = np.array(auc_diffs)
    print("AUC Differences:", auc_diffs)
    p_value = np.mean(auc_diffs <= 0)
    return p_value, auc_diffs

print(f"Mean AUC model 1 (TFIDF): {roc_auc_score(y_tfidf, _model_predict(model_TFIDF, X_tfidf)):.4f}")
print(f"Mean AUC model 2 (Count): {roc_auc_score(y_count, _model_predict(model_Count, X_count)):.4f}")

p_value, auc_diffs = bootstrap_auc_pvalue(X_tfidf, y_tfidf, model_TFIDF, X_count, y_count, model_Count, num_samples=1000)
print(f"P-value (AUC model 2 > model 1): {p_value:.10f}")