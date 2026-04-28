import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

print("\n" + "="*60)
print("  GENDER EQUALITY - FULL TRAINING PIPELINE")
print("="*60)

# =====================================================
# 1. LOAD DATA
# =====================================================
df = pd.read_csv("cleaned_gender_data.csv")
print("Dataset Shape:", df.shape)

# =====================================================
# 2. FEATURES
# =====================================================
feature_cols = [
    'Gender','Age','Department','Job_Level','Education',
    'Experience_Years','Years_at_Company','Annual_Salary',
    'Bonus','Performance_Rating','Training_Hours',
    'Parental_Leave_Taken','Leadership_Role'
]

target_col = "Promoted_Last_Year"

df = df[feature_cols + [target_col]].dropna()

# =====================================================
# 3. LABEL ENCODING
# =====================================================
label_encoders = {}
cat_cols = ['Gender','Department','Job_Level','Education','Performance_Rating']

for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le

# =====================================================
# 4. SPLIT DATA
# =====================================================
X = df[feature_cols]
y = df[target_col]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# =====================================================
# 5. SCALING
# =====================================================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# =====================================================
# 6. MODEL TRAINING
# =====================================================
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train_scaled, y_train)

# =====================================================
# 7. PREDICTION + EVALUATION
# =====================================================
y_pred = model.predict(X_test_scaled)

acc = accuracy_score(y_test, y_pred)
cv = cross_val_score(model, X_train_scaled, y_train, cv=5)

print("\nTest Accuracy:", round(acc, 4))
print("CV Accuracy:", round(cv.mean(), 4), "±", round(cv.std(), 4))

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# =====================================================
# 8. CONFUSION MATRIX (IMPORTANT)
# =====================================================
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Not Promoted','Promoted'],
            yticklabels=['Not Promoted','Promoted'])

plt.title("Confusion Matrix - Promotion Prediction")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.show()

print("✅ Confusion Matrix saved")

# =====================================================
# 9. FEATURE IMPORTANCE (VERY IMPORTANT FOR HR)
# =====================================================
feat_imp = pd.DataFrame({
    "Feature": feature_cols,
    "Importance": model.feature_importances_
}).sort_values("Importance", ascending=False)

plt.figure(figsize=(8,5))
sns.barplot(x="Importance", y="Feature", data=feat_imp)
plt.title("Feature Importance")
plt.tight_layout()
plt.savefig("feature_importance.png")
plt.show()

print("\nTop Features:\n")
print(feat_imp.head(10))

# =====================================================
# 10. SAVE MODEL ARTIFACTS (FOR FLASK)
# =====================================================
joblib.dump(model, "promotion_model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(feature_cols, "model_features.pkl")
joblib.dump(label_encoders, "label_encoders.pkl")

print("\n✅ ALL FILES SAVED SUCCESSFULLY")
print(" - promotion_model.pkl")
print(" - scaler.pkl")
print(" - model_features.pkl")
print(" - label_encoders.pkl")
print(" - confusion_matrix.png")
print(" - feature_importance.png")