import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.multioutput import MultiOutputClassifier

# Load dataset
df = pd.read_csv("mental_health_dataset.csv")

# Feature columns: Q1 to Q50, Label columns: last 9 columns
X = df.iloc[:, :-9]
y = df.iloc[:, -9:]

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
pickle.dump(scaler, open("scaler.pkl", "wb"))

# Base models
base_models = {
    "logistic_regression": LogisticRegression(max_iter=1000),
    "random_forest": RandomForestClassifier(),
    "svm": SVC(probability=True),
    "knn": KNeighborsClassifier(),
    "naive_bayes": GaussianNB(),
    "gradient_boosting": GradientBoostingClassifier()
}

# Wrap with MultiOutputClassifier for multi-label prediction
models = {name: MultiOutputClassifier(model) for name, model in base_models.items()}

# Train and save each model
for name, model in models.items():
    model.fit(X_scaled, y)
    with open(f"{name}.pkl", "wb") as f:
        pickle.dump(model, f)

print("✅ All models trained and saved successfully.")
