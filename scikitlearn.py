import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import make_scorer, precision_score, recall_score, f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.base import clone

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

# ------------------------------------------------------------
# 17. Define hyperparameter grid
# ------------------------------------------------------------

param_grid = {
    "model__C": [0.01, 0.1, 1, 10, 100],
    "model__penalty": ["l2"],
    "model__solver": ["lbfgs"]
}

# ------------------------------------------------------------
# 18. Build GridSearchCV
# ------------------------------------------------------------

grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    cv=cv,
    scoring=make_scorer(f1_score, pos_label=0, zero_division=0),
    n_jobs=-1,
    verbose=1
)

# ------------------------------------------------------------
# 19. Fit GridSearchCV on training data only
# ------------------------------------------------------------

grid_search.fit(X_train, y_train)

# ------------------------------------------------------------
# 20. Print best cross-validation results
# ------------------------------------------------------------

print("\nBest parameters:")
print(grid_search.best_params_)

print("\nBest cross-validation malignant F1:")
print(grid_search.best_score_)

# ------------------------------------------------------------
# 21. Evaluate best model on untouched test data
# ------------------------------------------------------------

best_model = grid_search.best_estimator_

y_pred_best = best_model.predict(X_test)

print("\nFinal test accuracy after GridSearchCV:")
print(accuracy_score(y_test, y_pred_best))

print("\nFinal test classification report after GridSearchCV:")
print(classification_report(
    y_test,
    y_pred_best,
    target_names=data.target_names
))

print("\nFinal test confusion matrix after GridSearchCV:")
print(confusion_matrix(y_test, y_pred_best, labels=[0, 1]))

# ------------------------------------------------------------
# 22. Inspect all GridSearchCV results
# ------------------------------------------------------------

grid_results = pd.DataFrame(grid_search.cv_results_)

important_columns = [
    "param_model__C",
    "mean_test_score",
    "std_test_score",
    "rank_test_score"
]

print("\nGridSearchCV results:")
print(
    grid_results[important_columns].sort_values("rank_test_score")
)

# ------------------------------------------------------------
# 23. Threshold analysis for malignant class
# ------------------------------------------------------------

class_order = best_model.named_steps["model"].classes_

print("\nClass order inside Logistic Regression:")
print(class_order)

malignant_class_index = list(class_order).index(0)

malignant_probabilities = best_model.predict_proba(X_test)[:, malignant_class_index]

thresholds = [0.30, 0.40, 0.50, 0.60, 0.70]

threshold_results = []

for threshold in thresholds:
    y_pred_threshold = np.where(
        malignant_probabilities >= threshold,
        0,  # predict malignant
        1   # predict benign
    )

    cm_threshold = confusion_matrix(
        y_test,
        y_pred_threshold,
        labels=[0, 1]
    )

    malignant_precision = precision_score(
        y_test,
        y_pred_threshold,
        pos_label=0,
        zero_division=0
    )

    malignant_recall = recall_score(
        y_test,
        y_pred_threshold,
        pos_label=0,
        zero_division=0
    )

    malignant_f1 = f1_score(
        y_test,
        y_pred_threshold,
        pos_label=0,
        zero_division=0
    )

    threshold_results.append({
        "threshold": threshold,
        "malignant_precision": malignant_precision,
        "malignant_recall": malignant_recall,
        "malignant_f1": malignant_f1,
        "missed_malignant_cases": cm_threshold[0, 1],
        "false_malignant_alarms": cm_threshold[1, 0],
    })


threshold_results_df = pd.DataFrame(threshold_results)

print("\nThreshold analysis:")
print(threshold_results_df.to_string(index=False))

# ------------------------------------------------------------
# 24. ROC-AUC for malignant class
# ------------------------------------------------------------

# sklearn's roc_auc_score expects the positive class to be 1.
# But in this dataset:
# 0 = malignant
# 1 = benign
#
# So we create a new binary target:
# malignant = 1
# benign = 0

y_test_malignant = (y_test == 0).astype(int)

# We already extracted probability of malignant class earlier:
# malignant_probabilities = best_model.predict_proba(X_test)[:, malignant_class_index]

roc_auc = roc_auc_score(
    y_test_malignant,
    malignant_probabilities
)

print("\nROC-AUC for malignant class:")
print(roc_auc)

# ------------------------------------------------------------
# 25. ROC curve values
# ------------------------------------------------------------

fpr, tpr, roc_thresholds = roc_curve(
    y_test_malignant,
    malignant_probabilities
)

roc_table = pd.DataFrame({
    "threshold": roc_thresholds,
    "tpr_recall": tpr,
    "fpr": fpr
})

print("\nFirst 10 ROC curve rows:")
print(roc_table.head(10).to_string(index=False))

# ------------------------------------------------------------
# 26. Evaluate a custom malignant threshold
# ------------------------------------------------------------

custom_threshold = 0.485

y_pred_custom = np.where(
    malignant_probabilities >= custom_threshold,
    0, #predict malignant
    1  #predict benign
)

print("\nCustom threshold:")
print(custom_threshold)

print("\nClassification report with custom threshold:")
print(classification_report(
    y_test,
    y_pred_custom,
    target_names=data.target_names
))

print("\nConfusion matrix with custom threshold:")
print(confusion_matrix(y_test, y_pred_custom, labels=[0, 1]))

# ------------------------------------------------------------
# 27. Plot ROC curve
# ------------------------------------------------------------

plt.figure(figsize=(8, 6))

plt.plot(fpr, tpr, label=f"Logistic Regression ROC-AUC = {roc_auc:.4f}")
plt.plot([0,1], [0, 1], linestyle="--", label="Random guessing")

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate / Recall")
plt.title("ROC Curve for Malignant Class")
plt.grid(True)

plt.show()

# ------------------------------------------------------------
# 28. Professional threshold selection using validation data
# ------------------------------------------------------------

# Split the original training data into:
# - inner training data
# - validation data
#
# The test set remains untouched.

X_train_inner, X_val, y_train_inner, y_val = train_test_split(
    X_train,
    y_train,
    test_size=0.2,
    random_state=42,
    stratify=y_train
)

# ------------------------------------------------------------
# 29. Fit GridSearchCV only on inner training data
# ------------------------------------------------------------

threshold_grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    cv=cv,
    scoring=make_scorer(f1_score, pos_label=0, zero_division=0),
    n_jobs=-1,
    verbose=1
)

threshold_grid_search.fit(X_train_inner, y_train_inner)

print("\nBest parameters from inner training data:")
print(threshold_grid_search.best_params_)

print("\nBest inner CV malignant F1:")
print(threshold_grid_search.best_score_)

# ------------------------------------------------------------
# 30. Get probabilities on validation data
# ------------------------------------------------------------

validation_model = threshold_grid_search.best_estimator_

class_order = validation_model.named_steps["model"].classes_

malignant_class_index = list(class_order).index(0)

val_malignant_probabilities = validation_model.predict_proba(X_val)[:, malignant_class_index]


# ------------------------------------------------------------
# 31. Test different thresholds on validation data
# ------------------------------------------------------------

threshold_candidates = np.arange(0.10, 0.91, 0.05)

validation_threshold_results = []

for threshold in threshold_candidates:
    y_val_pred_threshold = np.where(
        val_malignant_probabilities >= threshold,
        0,
        1
    )

    cm_val = confusion_matrix(
        y_val,
        y_val_pred_threshold,
        labels=[0, 1]
    )

    malignant_precision = precision_score(
        y_val,
        y_val_pred_threshold,
        pos_label=0,
        zero_division=0
    )

    malignant_recall = recall_score(
        y_val,
        y_val_pred_threshold,
        pos_label=0,
        zero_division=0
    )

    malignant_f1 = f1_score(
        y_val,
        y_val_pred_threshold,
        pos_label=0,
        zero_division=0
    )

    validation_threshold_results.append({
        "threshold": threshold,
        "malignant_precision": malignant_precision,
        "malignant_recall": malignant_recall,
        "malignant_f1": malignant_f1,
        "missed_malignant_cases": cm_val[0, 1],
        "false_malignant_alarms": cm_val[1, 0],
    })


validation_threshold_df = pd.DataFrame(validation_threshold_results)

print("\nValidation threshold analysis:")
print(validation_threshold_df.to_string(index=False))

# ------------------------------------------------------------
# 32. Choose threshold based on validation results
# ------------------------------------------------------------

# Strategy:
# First, look for thresholds with malignant recall >= 0.97.
# Among those, choose the one with the highest malignant precision.
# If no threshold reaches 0.97 recall, choose the highest malignant F1.

high_recall_thresholds = validation_threshold_df[
    validation_threshold_df["malignant_recall"] >= 0.97
]

if len(high_recall_thresholds) > 0:
    best_threshold_row = high_recall_thresholds.sort_values(
        by=["malignant_precision", "malignant_f1"],
        ascending=False
    ).iloc[0]
else:
    best_threshold_row = validation_threshold_df.sort_values(
        by="malignant_f1",
        ascending=False
    ).iloc[0]


selected_threshold = best_threshold_row["threshold"]

print("\nSelected threshold from validation data:")
print(selected_threshold)

print("\nSelected threshold row:")
print(best_threshold_row)

# ------------------------------------------------------------
# 33. Refit final model on full training data using best parameters
# ------------------------------------------------------------

final_model = clone(threshold_grid_search.best_estimator_)

final_model.fit(X_train, y_train)


# ------------------------------------------------------------
# 34. Apply selected threshold to test data
# ------------------------------------------------------------

final_class_order = final_model.named_steps["model"].classes_

final_malignant_class_index = list(final_class_order).index(0)

test_malignant_probabilities = final_model.predict_proba(X_test)[:, final_malignant_class_index]

y_test_pred_selected_threshold = np.where(
    test_malignant_probabilities >= selected_threshold,
    0,
    1
)


# ------------------------------------------------------------
# 35. Final test evaluation
# ------------------------------------------------------------

print("\nFinal test evaluation using validation-selected threshold:")

print("\nSelected threshold:")
print(selected_threshold)

print("\nClassification report:")
print(classification_report(
    y_test,
    y_test_pred_selected_threshold,
    target_names=data.target_names
))

print("\nConfusion matrix:")
print(confusion_matrix(
    y_test,
    y_test_pred_selected_threshold,
    labels=[0, 1]
))