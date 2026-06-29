import pandas as pd

from sklearn.datasets import load_breast_cancer

# ------------------------------------------------------------
# 1. Load the dataset from scikit-learn
# ------------------------------------------------------------

data = load_breast_cancer(as_frame=True)

# ------------------------------------------------------------
# 2. Separate features and target
# ------------------------------------------------------------

X = data.data
y = data.target

# ------------------------------------------------------------
# 3. Inspect the dataset
# ------------------------------------------------------------

print("Feature matrix shape:")
print(X.shape)

print("\nTarget shape:")
print(y.shape)

print("\nTarget names:")
print(data.target_names)

print("\nFeature names:")
print(data.feature_names)

print("\nFirst 5 rows of X:")
print(X.head())

print("\nFirst 5 values of y:")
print(y.head())

print("\nTarget value counts:")
print(y.value_counts())

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# ------------------------------------------------------------
# 4. Split data into training and test sets
# ------------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ------------------------------------------------------------
# 5. Build a proper scikit-learn Pipeline
# ------------------------------------------------------------

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(max_iter=5000, random_state=42))
])

# ------------------------------------------------------------
# 6. Train the pipeline on training data only
# ------------------------------------------------------------

pipeline.fit(X_train, y_train)

# ------------------------------------------------------------
# 7. Predict on test data
# ------------------------------------------------------------

y_pred = pipeline.predict(X_test)

# ------------------------------------------------------------
# 8. Evaluate the model
# ------------------------------------------------------------

accuracy = accuracy_score(y_test, y_pred)

print("\nTrain shape:")
print(X_train.shape)

print("\nTest shape:")
print(X_test.shape)

print("\nAccuracy:")
print(accuracy)

print("\nConfusion matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification report:")
print(classification_report(
    y_test,
    y_pred,
    target_names=data.target_names
))

# ------------------------------------------------------------
# 9. Make confusion matrix easier to understand
# ------------------------------------------------------------

cm = confusion_matrix(y_test, y_pred, labels=[0, 1])

cm_table = pd.DataFrame(
    cm,
    index=["Actual malignant", "Actual benign"],
    columns=["Predicted malignant", "Predicted benign"]
)

print("\nReadable confusion matrix:")
print(cm_table)

# ------------------------------------------------------------
# 10. Manually calculate accuracy
# ------------------------------------------------------------
correct_predictions = cm[0,0] + cm[1,1]
total_predictions = cm.sum()

manual_accuracy = correct_predictions / total_predictions

print("\nManual accuracy:")
print(manual_accuracy)

# ------------------------------------------------------------
# 11. Manually calculate malignant precision and recall
# ------------------------------------------------------------

malignant_precision = cm[0, 0] / (cm[0, 0] + cm[1, 0])
malignant_recall = cm[0, 0] / (cm[0, 0] + cm[0, 1])

print("\nMalignant precision:")
print(malignant_precision)

print("\nMalignant recall:")
print(malignant_recall)

# ------------------------------------------------------------
# 12. Manually calculate benign precision and recall
# ------------------------------------------------------------

benign_precision = cm[1, 1] / (cm[0, 1] + cm[1, 1])
benign_recall = cm[1, 1] / (cm[1, 0] + cm[1, 1])

print("\nBenign precision:")
print(benign_precision)

print("\nBenign recall:")
print(benign_recall)

from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import make_scorer, precision_score, recall_score, f1_score


# ------------------------------------------------------------
# 13. Cross-validation on training data only
# ------------------------------------------------------------

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


# ------------------------------------------------------------
# 14. Define scoring metrics
# ------------------------------------------------------------

scoring = {
    "accuracy": "accuracy",
    "malignant_precision": make_scorer(
        precision_score,
        pos_label=0,
        zero_division=0
    ),
    "malignant_recall": make_scorer(
        recall_score,
        pos_label=0,
        zero_division=0
    ),
    "malignant_f1": make_scorer(
        f1_score,
        pos_label=0,
        zero_division=0
    ),
}


# ------------------------------------------------------------
# 15. Run cross-validation
# ------------------------------------------------------------

cv_results = cross_validate(
    pipeline,
    X_train,
    y_train,
    cv=cv,
    scoring=scoring
)


# ------------------------------------------------------------
# 16. Print cross-validation results
# ------------------------------------------------------------

print("\nCross-validation results:")

for metric_name in scoring.keys():
    scores = cv_results[f"test_{metric_name}"]

    print(f"\n{metric_name}:")
    print("Fold scores:", scores)
    print("Mean:", scores.mean())
    print("Standard deviation:", scores.std())