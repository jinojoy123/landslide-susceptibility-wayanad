import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, classification_report
import joblib

df = pd.read_csv('combined_table_LR.csv').dropna()
X = df[['Curvature','Relief','Slope','TWI']].values
y = df['label'].values

# Standardize
scaler = StandardScaler().fit(X)
Xs = scaler.transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    Xs, y, test_size=0.3, stratify=y, random_state=42
)

# Weighted logistic regression
model = LogisticRegression(
    max_iter=1000,
    class_weight='balanced'
).fit(X_train, y_train)

y_scores = model.predict_proba(X_test)[:,1]
y_pred = (y_scores > 0.35).astype(int)  # custom threshold

auc = roc_auc_score(y_test, y_scores)
print("AUC:", auc)

print(classification_report(y_test, y_pred))

# Save
joblib.dump(model, 'logreg_model.joblib')
joblib.dump(scaler, 'logreg_scaler.joblib')