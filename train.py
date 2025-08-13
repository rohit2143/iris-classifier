import json
import os
import joblib
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Create models folder
os.makedirs("models", exist_ok=True)

# Load dataset
iris = datasets.load_iris()
X, y = iris.data, iris.target

# DataFrame for saving
df = pd.DataFrame(X, columns=iris.feature_names)
df["species"] = iris.target_names[y]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scaling
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

# Models
models = {
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "LogisticRegression": LogisticRegression(max_iter=500),
    "RandomForest": RandomForestClassifier(n_estimators=200, random_state=42),
    "SVM": SVC(kernel="rbf", probability=True, random_state=42),
}

# Train, evaluate, save
for name, clf in models.items():
    clf.fit(X_train_s, y_train)
    y_pred = clf.predict(X_test_s)

    acc = accuracy_score(y_test, y_pred)
    rep = classification_report(y_test, y_pred, target_names=iris.target_names, output_dict=True)
    cm = confusion_matrix(y_test, y_pred).tolist()

    # Save model
    joblib.dump(clf, f"models/{name}.pkl")

    # Save metrics
    metrics = {"accuracy": acc, "report": rep, "cm": cm}
    with open(f"models/{name}_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

# Save shared files
joblib.dump(scaler, "models/scaler.pkl")
joblib.dump(iris, "models/iris_dataset.pkl")
df.to_csv("models/iris_df.csv", index=False)

print("✅ Models trained and saved in 'models/' folder.")
