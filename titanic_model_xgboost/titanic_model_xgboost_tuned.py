import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set correct image output directory
image_dir = r"C:\Users\ri\OneDrive\ai project\model\titanic_model_xgboost\images"
if not os.path.exists(image_dir):
    os.makedirs(image_dir)

# Load preprocessed dataset
train_scaled = pd.read_csv(r"C:\Users\ri\OneDrive\ai project\model\titanic_model_xgboost\data\train_scaled.csv")

# Define features and target
X = train_scaled.drop(columns=['Survived', 'PassengerId'])
y = train_scaled['Survived']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define parameter grid for GridSearchCV
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.3]
}

# Initialize XGBoost Model
base_model = xgb.XGBClassifier(eval_metric='logloss')

# Perform GridSearchCV with 5-fold cross-validation
grid_search = GridSearchCV(estimator=base_model, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)

# Get best parameters and score
print("Best Parameters:", grid_search.best_params_)
print("Best Cross-Validation Accuracy:", grid_search.best_score_)

# Train new model with best parameters
tuned_model = xgb.XGBClassifier(**grid_search.best_params_, eval_metric='logloss')
tuned_model.fit(X_train, y_train)

# Predictions
y_pred = tuned_model.predict(X_test)

# Save predictions to CSV
predictions_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
predictions_df.to_csv(r"C:\Users\ri\OneDrive\ai project\model\titanic_model_xgboost\data\titanic_xgb_tuned_predictions.csv", index=False)

# Evaluation Metrics
print("\nTuned Model Classification Report:")
print(classification_report(y_test, y_pred))
print("Tuned Model Accuracy:", accuracy_score(y_test, y_pred))

# Visualize and save Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Tuned Model Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig(os.path.join(image_dir, "confusion_matrix_xgb_tuned.png"))
plt.show()
