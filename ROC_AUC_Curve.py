import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt

df = pd.read_csv("points_LSI_LR.csv")

y = df['label']
scores = df['LSI_value']

auc = roc_auc_score(y, scores)
print("ROC-AUC:", auc)

fpr, tpr, _ = roc_curve(y, scores)

plt.plot(fpr, tpr)
plt.plot([0,1],[0,1],'--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC-AUC Curve (LR Model)")
plt.show()
