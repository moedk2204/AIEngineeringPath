import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _nbutil import md, code, write

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "02-Supervised-Learning")

HEAD = '''
import pandas as pd, numpy as np
import matplotlib.pyplot as plt

pd.set_option("display.max_columns", 50)
np.random.seed(42)

heart = pd.read_csv("../data/heart-disease.csv")
X = heart.drop("target", axis=1)
y = heart["target"]
print(X.shape, y.value_counts().to_dict())
'''

# ---------------------------------------------------------------- 04
write(os.path.join(OUT, "04-cross-validation-and-leakage.ipynb"), [
md('''
# 04 - Cross-Validation & Data Leakage

Two ideas that decide whether your reported score is real.

1. **Cross-validation** - one train/test split gives you a number with a lot of luck in it.
2. **Data leakage** - the model sees information at training time that it would not have
   in the real world, so the score is optimistic and the model fails in production.

Notebook 01 used `Pipeline`. This notebook shows you *why* it exists.
'''),
code(HEAD),
md('''
## 1. A single split is a lottery ticket

Same model, same data, same code - only `random_state` on the split changes.
'''),
code('''
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

scores = []
for seed in range(20):
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=seed, stratify=y)
    m = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_tr, y_tr)
    scores.append(m.score(X_te, y_te))

scores = np.array(scores)
print(f"lowest  : {scores.min():.3f}")
print(f"highest : {scores.max():.3f}")
print(f"spread  : {scores.max() - scores.min():.3f}   <- pure luck of the split")
print(f"mean    : {scores.mean():.3f} +/- {scores.std():.3f}")
'''),
code('''
plt.figure(figsize=(9, 3.5))
plt.bar(range(20), scores, color="#4C78A8")
plt.axhline(scores.mean(), color="#E45756", ls="--", label=f"mean {scores.mean():.3f}")
plt.ylim(scores.min() - 0.05, scores.max() + 0.03)
plt.xlabel("random_state of the split"); plt.ylabel("test accuracy")
plt.title("The same model scored 20 different ways")
plt.legend()
plt.show()
'''),
md('''
> If you report the best of those, you are not reporting your model - you are reporting
> your luck. Cross-validation replaces one split with k splits and averages them.

## 2. K-Fold and Stratified K-Fold
'''),
code('''
from sklearn.model_selection import cross_val_score, KFold, StratifiedKFold

model = RandomForestClassifier(n_estimators=100, random_state=42)

cv_scores = cross_val_score(model, X, y, cv=5, scoring="accuracy")
print("fold scores:", np.round(cv_scores, 3))
print(f"mean {cv_scores.mean():.3f} +/- {cv_scores.std():.3f}")
'''),
md('''
### Stratify your folds for classification

`StratifiedKFold` keeps the class balance of the full dataset inside every fold.
For classification this is what you want, and `cross_val_score` already uses it by
default when you pass an integer `cv` and a classifier. Being explicit is still good habit.
'''),
code('''
plain = cross_val_score(model, X, y, cv=KFold(5, shuffle=True, random_state=42))
strat = cross_val_score(model, X, y, cv=StratifiedKFold(5, shuffle=True, random_state=42))

print(f"KFold           : {plain.mean():.3f} +/- {plain.std():.3f}")
print(f"StratifiedKFold : {strat.mean():.3f} +/- {strat.std():.3f}")
'''),
md('''
### Choosing a CV splitter

| Splitter | Use when |
|---|---|
| `KFold` | plain regression, rows are independent |
| `StratifiedKFold` | **classification** - the default you want |
| `GroupKFold` | rows are grouped (several visits per patient, several photos per user) and a group must not straddle the split |
| `TimeSeriesSplit` | the data has a time order - never train on the future |

Getting this wrong is itself a form of leakage. If the same patient appears in both train
and test, the model can recognise the patient rather than learn the disease.

## 3. More than one metric at a time
'''),
code('''
from sklearn.model_selection import cross_validate

results = cross_validate(
    model, X, y, cv=5,
    scoring=["accuracy", "precision", "recall", "f1", "roc_auc"],
    return_train_score=True)

summary = pd.DataFrame({
    k.replace("test_", ""): [v.mean(), v.std()]
    for k, v in results.items() if k.startswith("test_")
}, index=["mean", "std"]).T.round(3)
summary
'''),
md('''
## 4. Data leakage - the expensive mistake

Here is the wrong way to scale, which almost every beginner writes at least once:

```python
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)          # <- fitted on ALL the data
X_train, X_test = train_test_split(X_scaled)
```

The scaler computed the mean and standard deviation using the test rows. Information from
the test set has leaked into training. The reported score goes up; real-world performance
does not.

Let us measure the size of the lie. To make it visible, we use a feature-selection step,
where leakage bites hardest.
'''),
code('''
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

# Random noise features - there is genuinely NOTHING to learn here.
rng = np.random.RandomState(0)
n_samples = 200
X_noise = pd.DataFrame(rng.randn(n_samples, 2000))
y_noise = pd.Series(rng.randint(0, 2, n_samples))

# --- WRONG: select features using the whole dataset, then cross-validate ---
selector = SelectKBest(f_classif, k=20).fit(X_noise, y_noise)
X_selected = selector.transform(X_noise)
leaky = cross_val_score(LogisticRegression(max_iter=1000), X_selected, y_noise, cv=5)

# --- RIGHT: selection happens inside the pipeline, refit on each fold ---
honest_pipe = Pipeline([
    ("select", SelectKBest(f_classif, k=20)),
    ("clf", LogisticRegression(max_iter=1000)),
])
honest = cross_val_score(honest_pipe, X_noise, y_noise, cv=5)

print(f"LEAKY  score on pure noise : {leaky.mean():.3f}")
print(f"HONEST score on pure noise : {honest.mean():.3f}")
print("\\nTruth: the data is random. Anything above ~0.50 is an illusion.")
'''),
md('''
That is the whole lesson. The leaky version invents a model that looks useful and is
worth nothing. The pipeline version correctly reports "there is no signal here".

### The rule

> **Every step that learns something from the data - imputers, scalers, encoders,
> feature selectors, PCA - must be fitted inside the cross-validation loop.**
> A `Pipeline` guarantees that for free.
'''),
code('''
# The correct pattern, end to end.
correct = Pipeline([
    ("scale", StandardScaler()),
    ("clf", LogisticRegression(max_iter=1000)),
])

cv_scores = cross_val_score(correct, X, y, cv=StratifiedKFold(5, shuffle=True, random_state=42))
print(f"honest cross-validated accuracy: {cv_scores.mean():.3f} +/- {cv_scores.std():.3f}")
'''),
md('''
## 5. Other leakage traps to recognise

- **A target-derived feature.** Predicting churn using `cancellation_date`. The column is
  only filled in *after* the event you are predicting.
- **Duplicate rows** split across train and test.
- **Time travel.** Using next week's data to predict this week. Always `TimeSeriesSplit`.
- **Preprocessing before splitting.** The case demonstrated above.

Sanity check: if your accuracy jumps to 0.99 on a hard problem, assume leakage and go
looking for it before you celebrate.

---
## Exercises

1. Re-run the 20-seed experiment with `LogisticRegression` inside a scaling pipeline.
   Is the spread bigger or smaller than the Random Forest's? Why might that be?
2. Change `cv=5` to `cv=10` in `cross_val_score`. What happens to the mean, and what
   happens to the standard deviation? What is the cost?
3. Build the leaky-vs-honest comparison again with `k=100` selected noise features.
   Does the illusion get stronger or weaker?
4. Write a pipeline for `car-sales-extended-missing-data.csv` that imputes, encodes and
   scales, then report a 5-fold cross-validated R^2. Confirm no step touches the test fold.
5. **Stretch:** invent a leaky feature for `heart-disease.csv` - for example
   `target + small noise` - add it as a column and watch cross-validated accuracy hit
   ~1.00. Then explain how you would have caught it in a code review.
'''),
])

# ---------------------------------------------------------------- 05
write(os.path.join(OUT, "05-classification-metrics.ipynb"), [
md('''
# 05 - Classification Metrics & Decision Thresholds

Notebook 01 printed a `classification_report` without saying what to do with it.
Accuracy is the wrong metric more often than beginners expect, and the default
0.5 decision threshold is a choice, not a law.

By the end you will be able to:
1. Explain why accuracy lies on imbalanced data.
2. Read a confusion matrix and choose between precision and recall.
3. Move the decision threshold on purpose.
4. Pick between ROC-AUC and Average Precision.
5. Handle class imbalance with `class_weight`.
'''),
code(HEAD),
md('''
## 1. The accuracy trap

Build a deliberately imbalanced problem: 97% negative, 3% positive - fraud, rare disease,
machine failure. All the interesting problems look like this.
'''),
code('''
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression

X_imb, y_imb = make_classification(
    n_samples=5000, n_features=20, n_informative=4,
    weights=[0.97, 0.03], random_state=42)

Xi_tr, Xi_te, yi_tr, yi_te = train_test_split(
    X_imb, y_imb, test_size=0.3, random_state=42, stratify=y_imb)

print("class balance:", np.bincount(y_imb), f"({y_imb.mean():.1%} positive)")

lazy = DummyClassifier(strategy="most_frequent").fit(Xi_tr, yi_tr)
print(f"\\nA model that always predicts 'no': accuracy = {lazy.score(Xi_te, yi_te):.3f}")
print("It catches exactly zero positives. Accuracy is useless here.")
'''),
md('''
## 2. The confusion matrix is the source of truth

Every other classification metric is a ratio computed from these four numbers.

|  | predicted 0 | predicted 1 |
|---|---|---|
| **actual 0** | TN | FP (false alarm) |
| **actual 1** | FN (missed case) | TP |
'''),
code('''
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

clf = LogisticRegression(max_iter=2000).fit(Xi_tr, yi_tr)
y_pred = clf.predict(Xi_te)

cm = confusion_matrix(yi_te, y_pred)
tn, fp, fn, tp = cm.ravel()
print(f"true negatives  {tn:5d}   false positives {fp:5d}  (false alarms)")
print(f"false negatives {fn:5d}   true positives  {tp:5d}  (caught)")

ConfusionMatrixDisplay(cm, display_labels=["no", "yes"]).plot(cmap="Blues", colorbar=False)
plt.title("Confusion matrix")
plt.show()
'''),
code('''
print(classification_report(yi_te, y_pred, target_names=["no", "yes"], digits=3, zero_division=0))
'''),
md('''
### What each row of that report means

- **precision** = TP / (TP + FP) - *of everything I flagged, how much was real?*
  Optimise this when a false alarm is expensive (spam filter deleting a real email).
- **recall** = TP / (TP + FN) - *of everything real, how much did I catch?*
  Optimise this when a miss is expensive (cancer screening, fraud detection).
- **f1** - the harmonic mean of the two. A single number when both matter.
- **support** - how many real examples of that class exist. Always check it: a precision
  of 1.00 on a support of 3 means nothing.

**You cannot maximise both.** Raise recall and you flag more things, so precision falls.
The business decides which error is worse, not the model.

## 3. Moving the threshold

`.predict()` silently applies `probability >= 0.5`. That 0.5 is a default, not a truth.
'''),
code('''
probs = clf.predict_proba(Xi_te)[:, 1]   # probability of class 1

from sklearn.metrics import precision_score, recall_score, f1_score

rows = []
for t in [0.10, 0.20, 0.30, 0.50, 0.70, 0.90]:
    pred_t = (probs >= t).astype(int)
    rows.append({
        "threshold": t,
        "flagged": int(pred_t.sum()),
        "precision": precision_score(yi_te, pred_t, zero_division=0),
        "recall": recall_score(yi_te, pred_t),
        "f1": f1_score(yi_te, pred_t),
    })
pd.DataFrame(rows).round(3)
'''),
code('''
from sklearn.metrics import precision_recall_curve

prec, rec, thr = precision_recall_curve(yi_te, probs)

plt.figure(figsize=(9, 4.5))
plt.plot(thr, prec[:-1], label="precision", color="#4C78A8")
plt.plot(thr, rec[:-1], label="recall", color="#F58518")
plt.axvline(0.5, ls="--", c="grey", label="default threshold 0.5")
plt.xlabel("decision threshold"); plt.ylabel("score")
plt.title("Lower the threshold: recall rises, precision falls")
plt.legend(); plt.grid(alpha=.3)
plt.show()
'''),
md('''
> **Practical use.** "We must catch at least 80% of fraud" is a recall target. Read the
> threshold off this plot that delivers it, then check what precision you have to live
> with. That is an engineering conversation, not a modelling one.

## 4. ROC-AUC vs Average Precision

Both summarise a model across *all* thresholds, so you can compare models before
committing to one.
'''),
code('''
from sklearn.metrics import roc_curve, roc_auc_score, average_precision_score

fpr, tpr, _ = roc_curve(yi_te, probs)
auc = roc_auc_score(yi_te, probs)
ap = average_precision_score(yi_te, probs)

fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

axes[0].plot(fpr, tpr, color="#4C78A8", lw=2, label=f"ROC-AUC = {auc:.3f}")
axes[0].plot([0, 1], [0, 1], "--", c="grey", label="random guessing")
axes[0].set_xlabel("false positive rate"); axes[0].set_ylabel("true positive rate")
axes[0].set_title("ROC curve"); axes[0].legend(); axes[0].grid(alpha=.3)

axes[1].plot(rec, prec, color="#F58518", lw=2, label=f"Avg precision = {ap:.3f}")
axes[1].axhline(yi_te.mean(), ls="--", c="grey", label=f"baseline = {yi_te.mean():.3f}")
axes[1].set_xlabel("recall"); axes[1].set_ylabel("precision")
axes[1].set_title("Precision-Recall curve"); axes[1].legend(); axes[1].grid(alpha=.3)

plt.tight_layout()
plt.show()
'''),
md('''
**Which one?**

- **ROC-AUC** - use when the classes are roughly balanced. It is optimistic on rare
  positives, because a huge TN count keeps the false positive rate low no matter what.
- **Average Precision (PR-AUC)** - use when positives are rare. Its baseline is the
  positive rate itself, so it is honest about hard problems.

Note above how the ROC-AUC looks excellent while average precision is far more sober.
Same model, same predictions.

## 5. Telling the model the classes are imbalanced
'''),
code('''
plain = LogisticRegression(max_iter=2000).fit(Xi_tr, yi_tr)
weighted = LogisticRegression(max_iter=2000, class_weight="balanced").fit(Xi_tr, yi_tr)

for name, m in [("default", plain), ("class_weight='balanced'", weighted)]:
    p = m.predict(Xi_te)
    print(f"{name:26} | precision {precision_score(yi_te, p, zero_division=0):.3f} "
          f"| recall {recall_score(yi_te, p):.3f} | f1 {f1_score(yi_te, p):.3f}")
'''),
md('''
`class_weight="balanced"` makes each mistake on the rare class count more during training.
Recall goes up, precision goes down - the same trade as moving the threshold, applied at
fit time instead of predict time.

---
## Exercises

1. Change `weights=[0.97, 0.03]` to `[0.5, 0.5]`. Do ROC-AUC and average precision still
   disagree? Explain what changed.
2. Find the threshold that gives at least **90% recall**. What precision do you get, and
   how many false alarms per 1000 cases is that?
3. Run the whole notebook on `heart-disease.csv` instead. Is it imbalanced? Which metric
   would you report to a doctor, and why?
4. Compare `class_weight="balanced"` against simply setting the threshold to 0.2. Do they
   land in the same place? What is different about them in practice?
5. **Stretch:** write `best_threshold(y_true, probs, min_recall=0.8)` that returns the
   highest-precision threshold meeting a recall floor. Use it to pick an operating point.
'''),
])
