print("Starting model training...")
import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
# Load dataset
df = pd.read_csv("data/Churn_Modelling.csv")
# Drop useless columns
df = df.drop(columns=["RowNumber", "CustomerId", "Surname"])
# Split features and target
X = df.drop(columns=["Exited"])
y = df["Exited"]
categorical_features = ["Geography", "Gender"]
numeric_features = [
   "CreditScore",
   "Age",
   "Tenure",
   "Balance",
   "NumOfProducts",
   "HasCrCard",
   "IsActiveMember",
   "EstimatedSalary"
]
preprocessor = ColumnTransformer(
   transformers=[
       ("num", StandardScaler(), numeric_features),
       ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), categorical_features)
   ]
)
scale_pos_weight = (y == 0).sum() / (y == 1).sum()
model = Pipeline(steps=[
   ("preprocessor", preprocessor),
   ("classifier", XGBClassifier(
       n_estimators=200,
       max_depth=4,
       learning_rate=0.1,
       scale_pos_weight=scale_pos_weight,
       random_state=42,
       eval_metric="logloss"
   ))
])
model.fit(X, y)
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/model.pkl")
print("Model trained and saved to models/model.pkl")
