import pandas as pd
import xgboost as xgb
import sklearn
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

# Load preprocessed dataset
train_scaled = pd.read_csv(r"C:\Users\ri\OneDrive\ai project\model\titanic_model_xgboost\data\train_scaled.csv")

# Define features and target
X = train_scaled.drop(columns=['Survived', 'PassengerId'])
y = train_scaled['Survived']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize XGBoost Model
model = xgb.XGBClassifier(eval_metric='logloss')

# Model Training
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Save predictions to CSV
predictions_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
predictions_df.to_csv(r"C:\Users\ri\OneDrive\ai project\model\titanic_model_xgboost\data\predictions.csv", index=False)

# Evaluation Metrics
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Visualize Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()