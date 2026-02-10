import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pickle

# Load dataset
data = pd.read_csv("dataset/Crop_recommendation.csv")

# Inputs and output
X = data.drop("label", axis=1)
y = data["label"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("="*60)
print("Training Multiple Machine Learning Models")
print("="*60)

# 1. Random Forest Classifier
print("\n1. Random Forest Classifier:")
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)
rf_accuracy = accuracy_score(y_test, rf_pred)
print(f"   Accuracy: {rf_accuracy * 100:.2f}%")

# 2. K-Nearest Neighbors (KNN)
print("\n2. K-Nearest Neighbors (KNN):")
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train, y_train)
knn_pred = knn_model.predict(X_test)
knn_accuracy = accuracy_score(y_test, knn_pred)
print(f"   Accuracy: {knn_accuracy * 100:.2f}%")

# 3. Logistic Regression
print("\n3. Logistic Regression:")
lr_model = LogisticRegression(max_iter=1000, random_state=42)
lr_model.fit(X_train, y_train)
lr_pred = lr_model.predict(X_test)
lr_accuracy = accuracy_score(y_test, lr_pred)
print(f"   Accuracy: {lr_accuracy * 100:.2f}%")

print("\n" + "="*60)
print("Model Summary:")
print("="*60)
print(f"Random Forest:       {rf_accuracy * 100:.2f}%")
print(f"KNN (k=5):          {knn_accuracy * 100:.2f}%")
print(f"Logistic Regression: {lr_accuracy * 100:.2f}%")

# Save the best model
models = {
    "Random Forest": (rf_model, rf_accuracy),
    "KNN": (knn_model, knn_accuracy),
    "Logistic Regression": (lr_model, lr_accuracy)
}

best_model_name = max(models, key=lambda x: models[x][1])
best_model, best_accuracy = models[best_model_name]

print(f"\nBest Model: {best_model_name} with {best_accuracy * 100:.2f}% accuracy")
print("="*60)

# Save best model
with open("model/crop_model.pkl", "wb") as f:
    pickle.dump(best_model, f)

# Save all models for comparison
with open("model/all_models.pkl", "wb") as f:
    pickle.dump(models, f)

print("\nAll models trained and saved successfully!")
