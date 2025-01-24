from matplotlib import pyplot as plt
from sklearn.datasets import load_digits
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.base import clone
import numpy as np

# ZAD 1
data = load_digits()
X, y = data.data, data.target

hidden_layer_configs = [(10,), (10, 10), (10, 10, 10), (10, 10, 10, 10), (10, 10, 10, 10, 10)]

cv = RepeatedStratifiedKFold(n_splits=2, n_repeats=5, random_state=42)

mean_results = []

for config in hidden_layer_configs:
    clf = MLPClassifier(hidden_layer_sizes=config, random_state=42, max_iter=1000)

    scores = cross_val_score(clone(clf), X, y, cv=cv, scoring='balanced_accuracy')
    mean_results.append(np.mean(scores))

mean_results = np.array(mean_results)
best_classifier = np.argmax(mean_results)
print("Mean results:", mean_results)
print("Argmax:", best_classifier)

# ZAD 2
known_classes = [0, 1, 2, 3, 4]
unknown_classes = [5, 6, 7, 8, 9]

best_hidden_layer_config = hidden_layer_configs[best_classifier]
clf = MLPClassifier(hidden_layer_sizes=best_hidden_layer_config, random_state=42, max_iter=1000)
cv = RepeatedStratifiedKFold(n_splits=2, n_repeats=5, random_state=42)

inner_scores = []
outer_scores = []

for train_idx, test_idx in cv.split(X, y):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    known_train_mask = np.isin(y_train, known_classes)
    X_train_known = X_train[known_train_mask]
    y_train_known = y_train[known_train_mask]

    clf_known: MLPClassifier = clone(clf)
    clf_known.fit(X_train_known, y_train_known)

    known_test_mask = np.isin(y_test, known_classes)
    X_test_known = X_test[known_test_mask]
    y_test_known = y_test[known_test_mask]

    y_pred_known = clf_known.predict(X_test_known)
    inner_score = balanced_accuracy_score(y_test_known, y_pred_known)
    inner_scores.append(inner_score)

    unknown_test_mask = np.isin(y_test, unknown_classes)
    X_test_unknown = X_test[unknown_test_mask]

    X_mixed = np.vstack((X_test_known, X_test_unknown))
    y_mixed = np.hstack((np.ones(len(X_test_known)), np.zeros(len(X_test_unknown))))

    y_proba = clf_known.predict_proba(X_mixed)
    decision_support = np.max(y_proba, axis=1)

    threshold = 0.8
    y_pred_mixed = (decision_support >= threshold).astype(int)

    outer_score = balanced_accuracy_score(y_mixed, y_pred_mixed)
    outer_scores.append(outer_score)

print("Inner score:", np.mean(inner_scores))
print("Outer score:", np.mean(outer_scores))

# ZAD 3
thresholds = np.linspace(0.5, 1, 100)
threshold_exp = False

decision_support_all = []
y_mixed_all = []

for train_idx, test_idx in cv.split(X, y):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    known_train_mask = np.isin(y_train, known_classes)
    X_train_known = X_train[known_train_mask]
    y_train_known = y_train[known_train_mask]

    clf_known: MLPClassifier = clone(clf)
    clf_known.fit(X_train_known, y_train_known)

    known_test_mask = np.isin(y_test, known_classes)
    X_test_known = X_test[known_test_mask]
    y_test_known = y_test[known_test_mask]

    y_pred_known = clf_known.predict(X_test_known)
    inner_score = balanced_accuracy_score(y_test_known, y_pred_known)
    inner_scores.append(inner_score)

    unknown_test_mask = np.isin(y_test, unknown_classes)
    X_test_unknown = X_test[unknown_test_mask]

    X_mixed = np.vstack((X_test_known, X_test_unknown))
    y_mixed = np.hstack((np.ones(len(X_test_known)), np.zeros(len(X_test_unknown))))

    y_proba = clf_known.predict_proba(X_mixed)
    decision_support = np.max(y_proba, axis=1)

    decision_support_all.extend(decision_support)
    y_mixed_all.extend(y_mixed)

outer_scores_by_threshold = []

for threshold in thresholds:
    y_pred_mixed = (np.array(decision_support_all) >= threshold).astype(int)
    outer_score = balanced_accuracy_score(np.array(y_mixed_all), y_pred_mixed)
    outer_scores_by_threshold.append(outer_score)

best_threshold_idx = np.argmax(outer_scores_by_threshold)
best_threshold = thresholds[best_threshold_idx]
best_score = outer_scores_by_threshold[best_threshold_idx]

print(f"Best threshold: {best_threshold}")
print(f"Best score: {best_score}")

plt.figure(figsize=(10, 6))
plt.plot(thresholds, outer_scores_by_threshold, label="Balanced Accuracy")
plt.scatter([best_threshold], [best_score], color='red')
plt.xlabel("Threshold")
plt.ylabel("Outer BAC")
plt.grid()
plt.show()
