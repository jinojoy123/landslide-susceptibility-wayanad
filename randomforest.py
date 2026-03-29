import pandas as pd 
import numpy as np 
from sklearn.ensemble import RandomForestClassifier 
from sklearn.model_selection import train_test_split, GridSearchCV 
from sklearn.metrics import roc_auc_score, classification_report 
import joblib

df = pd.read_csv('combined_table_RF.csv').dropna()
X = df[['Curvature','Relief','Slope','TWI']].values
y = df['label'].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.3, random_state=42)

#Hypertuning
# param_grid = {
#  'n_estimators':[100,200,400],
#  'max_depth':[None,10,20],
#  'min_samples_leaf':[1,2,4]
# }

# gs = GridSearchCV(
#     RandomForestClassifier(class_weight='balanced', n_jobs=-1, random_state=42),
#     param_grid,
#     scoring='roc_auc',
#     cv=5,
#     n_jobs=-1
# )

# gs.fit(X_train, y_train)
# print(gs.best_params_, gs.best_score_)

#Training
rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train, y_train)

probs = rf.predict_proba(X_test)[:,1]
print("RF AUC:", roc_auc_score(y_test, probs))
print(classification_report(y_test, rf.predict(X_test)))

fi = pd.Series(rf.feature_importances_, index=['Curvature','Relief','Slope','TWI']).sort_values(ascending=False)
print(fi)
joblib.dump(rf, 'rf_model.joblib')
