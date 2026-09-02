import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _nbutil import md, code, write

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "02-Supervised-Learning")

HEART = '''
import pandas as pd, numpy as np
import matplotlib.pyplot as plt

pd.set_option("display.max_columns", 50)
np.random.seed(42)

heart = pd.read_csv("../data/heart-disease.csv")
X = heart.drop("target", axis=1)
y = heart["target"]
print(X.shape, y.value_counts().to_dict())
'''

# ---------------------------------------------------------------- 06
write(os.path.join(OUT, "06-regression-and-regularization.ipynb"), [
md('''
# 06 - Regression Metrics & Regularization

Notebook 01 used `Ridge` without explaining what the "ridge" is. This notebook fixes that.

By the end you will be able to:
1. Choose between MAE, RMSE and R^2 and say what each one punishes.
2. Explain what regularization actually does to the coefficients.
3. Use `Ridge` (L2) and `Lasso` (L1), and know when each is right.
4. Use Lasso to drop useless features automatically.
'''),
code('''
import pandas as pd, numpy as np
import matplotlib.pyplot as plt
np.random.seed(42)

cars = pd.read_csv("../data/car-sales-extended-missing-data.csv").dropna(subset=["Price"])
X = cars.drop("Price", axis=1)
y = cars["Price"]
print(X.shape)
cars.head()
'''),
md('''
## 1. The three regression metrics
'''),
code('''
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

categorical, numeric = ["Make", "Colour"], ["Odometer (KM)", "Doors"]

prep = ColumnTransformer([
    ("cat", Pipeline([("i", SimpleImputer(strategy="constant", fill_value="missing")),
                      ("e", OneHotEncoder(handle_unknown="ignore"))]), categorical),
    ("num", Pipeline([("i", SimpleImputer(strategy="median")),
                      ("s", StandardScaler())]), numeric),
])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

lin = Pipeline([("prep", prep), ("model", LinearRegression())]).fit(X_train, y_train)
pred = lin.predict(X_test)

mae = mean_absolute_error(y_test, pred)
rmse = np.sqrt(mean_squared_error(y_test, pred))
r2 = r2_score(y_test, pred)

print(f"MAE  : {mae:9,.0f}   average error, in dollars")
print(f"RMSE : {rmse:9,.0f}   same units, but punishes big misses harder")
print(f"R^2  : {r2:9.3f}   fraction of variance explained (1.0 = perfect, 0 = no better than the mean)")
'''),
md('''
### Choosing between them

| Metric | Units | Behaviour | Use when |
|---|---|---|---|
| **MAE** | same as target | treats all errors equally | outliers are noise you want ignored |
| **RMSE** | same as target | squares errors, so one huge miss dominates | a large error is disproportionately costly |
| **R^2** | unitless | 1.0 perfect, 0 = predicting the mean, can go negative | comparing across different datasets |

**Report MAE to a stakeholder** - "we are off by about $4,000 on average" is a sentence
anyone understands. R^2 is for you, not for them.
'''),
code('''
# Why RMSE > MAE always: one big miss.
errors = np.abs(y_test - pred)
plt.figure(figsize=(9, 4))
plt.hist(errors, bins=40, color="#4C78A8")
plt.axvline(mae, color="#F58518", ls="--", lw=2, label=f"MAE {mae:,.0f}")
plt.axvline(rmse, color="#E45756", ls="--", lw=2, label=f"RMSE {rmse:,.0f}")
plt.xlabel("absolute error ($)"); plt.ylabel("count")
plt.title("RMSE sits to the right of MAE - the tail is pulling it")
plt.legend()
plt.show()
'''),
md('''
## 2. What regularization actually does

Regularization adds a penalty on the **size of the coefficients** to the loss function.
The model must now trade fitting the data against keeping its weights small.

- **Ridge (L2)** penalises the sum of *squared* coefficients. It shrinks everything
  towards zero, but never exactly to zero.
- **Lasso (L1)** penalises the sum of *absolute* coefficients. It can push a coefficient
  to exactly zero, which removes the feature.

`alpha` controls the strength. `alpha=0` is plain linear regression. Large `alpha` means
a simpler, flatter model.

Let us watch it happen on data designed to overfit.
'''),
code('''
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge, Lasso
from sklearn.pipeline import make_pipeline

rng = np.random.RandomState(0)
xs = np.sort(rng.rand(25)) * 6
ys = np.sin(xs) + rng.randn(25) * 0.3
grid = np.linspace(0, 6, 300).reshape(-1, 1)

fig, axes = plt.subplots(1, 4, figsize=(18, 3.8))
for ax, alpha in zip(axes, [0, 0.01, 1.0, 100.0]):
    model = (make_pipeline(PolynomialFeatures(15), LinearRegression()) if alpha == 0
             else make_pipeline(PolynomialFeatures(15), StandardScaler(), Ridge(alpha=alpha)))
    model.fit(xs.reshape(-1, 1), ys)
    ax.scatter(xs, ys, s=22, color="#333333", zorder=3)
    ax.plot(grid, model.predict(grid), color="#E45756", lw=2)
    ax.plot(grid, np.sin(grid.ravel()), "--", color="grey", lw=1)
    ax.set_ylim(-2.2, 2.2)
    ax.set_title("no penalty" if alpha == 0 else f"Ridge alpha={alpha}")
plt.suptitle("Degree-15 polynomial, increasing regularization ->")
plt.tight_layout()
plt.show()
'''),
md('''
Left: wild swings, the classic overfit. Right: over-regularised, almost a flat line -
we have regularised the signal away too. The second and third panels are the useful range.

**`alpha` is a hyperparameter. You tune it with cross-validation (notebook 08), never by eye.**

## 3. Ridge vs Lasso on real coefficients
'''),
code('''
# Build a dataset where only 5 of 40 features actually matter.
from sklearn.datasets import make_regression

Xr, yr, true_coef = make_regression(
    n_samples=200, n_features=40, n_informative=5,
    noise=15, coef=True, random_state=42)

Xr_tr, Xr_te, yr_tr, yr_te = train_test_split(Xr, yr, test_size=0.3, random_state=42)

models = {
    "LinearRegression": LinearRegression(),
    "Ridge (alpha=10)": Ridge(alpha=10),
    "Lasso (alpha=5)": Lasso(alpha=5, max_iter=10000),
}

fitted = {}
for name, m in models.items():
    m.fit(Xr_tr, yr_tr)
    fitted[name] = m
    print(f"{name:20} test R^2 = {m.score(Xr_te, yr_te):.3f} | "
          f"non-zero coefficients = {(np.abs(m.coef_) > 1e-6).sum():2d} / 40")
print(f"{'TRUTH':20} non-zero coefficients = {(true_coef != 0).sum():2d} / 40")
'''),
code('''
fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
for ax, (name, m) in zip(axes, fitted.items()):
    ax.bar(range(40), m.coef_, color="#4C78A8")
    ax.scatter(range(40), true_coef, color="#E45756", s=14, zorder=3, label="true coefficient")
    ax.set_ylabel(name.split(" ")[0]); ax.set_title(name, fontsize=10); ax.legend(fontsize=8)
axes[-1].set_xlabel("feature index")
plt.suptitle("Lasso zeroes out the 35 useless features; Ridge only shrinks them")
plt.tight_layout()
plt.show()
'''),
md('''
### The choice

| | Ridge (L2) | Lasso (L1) |
|---|---|---|
| Coefficients | shrunk, never zero | many become exactly zero |
| Keeps all features? | yes | no - it selects |
| Correlated features | shares weight between them | picks one arbitrarily |
| Use when | all features plausibly matter | you suspect most are useless |

`ElasticNet` mixes both and is a reasonable default when you cannot decide.

**Always scale before regularising.** The penalty is on coefficient size, and a feature
measured in millions gets an artificially tiny coefficient - so it escapes the penalty.

---
## Exercises

1. Compute MAE, RMSE and R^2 for a `RandomForestRegressor` on the car data. Which metric
   shows the biggest improvement over linear regression, and why?
2. Add one artificial outlier row to the car data with `Price = 500000`. Re-fit and watch
   MAE and RMSE. Which one moved more? Which model would you trust now?
3. Sweep `alpha` for Ridge over `np.logspace(-3, 3, 20)` on the `make_regression` data and
   plot test R^2. Where is the peak?
4. Do the same sweep for Lasso and plot the number of non-zero coefficients against alpha.
5. **Stretch:** use `RidgeCV` and `LassoCV`, which tune alpha internally by cross-validation.
   Compare their chosen alpha to your manual sweep.
'''),
])

# ---------------------------------------------------------------- 07
write(os.path.join(OUT, "07-trees-forests-and-boosting.ipynb"), [
md('''
# 07 - Decision Trees, Random Forests & Gradient Boosting

Notebook 01 used `RandomForestClassifier` as a black box. This notebook opens it.

Tree models are the workhorse of tabular machine learning. On spreadsheet-shaped data
a well-tuned gradient boosting model beats a neural network most of the time.

By the end you will be able to:
1. Explain how a tree picks a split.
2. Read `feature_importances_` - and know its trap.
3. Explain **bagging** (Random Forest) vs **boosting** (Gradient Boosting).
4. Choose between them.
'''),
code(HEART),
md('''
## 1. A single decision tree

A tree asks yes/no questions. At each node it tries every feature and every threshold,
and keeps the split that makes the two resulting groups **purest** - most one-sided in
their class labels. Purity is measured by *gini impurity* or *entropy*.
'''),
code('''
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier, plot_tree

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

tree = DecisionTreeClassifier(max_depth=3, random_state=42).fit(X_train, y_train)
print(f"train {tree.score(X_train, y_train):.3f} | test {tree.score(X_test, y_test):.3f}")

plt.figure(figsize=(20, 8))
plot_tree(tree, feature_names=X.columns, class_names=["no disease", "disease"],
          filled=True, rounded=True, fontsize=9)
plt.title("A depth-3 decision tree - readable by a human, which is the point")
plt.show()
'''),
md('''
**Read one box.** `gini` is the impurity (0 = perfectly pure), `samples` is how many
training rows reached this node, `value` is the class breakdown. The tree stops splitting
when it hits `max_depth`, or a node is pure, or too few samples remain.

This interpretability is the tree's superpower. You can print it and hand it to a doctor.

### The weakness: instability
'''),
code('''
# Change the training data slightly and the tree can restructure completely.
for seed in [0, 1, 2]:
    Xs, _, ys, _ = train_test_split(X, y, test_size=0.2, random_state=seed, stratify=y)
    t = DecisionTreeClassifier(max_depth=3, random_state=42).fit(Xs, ys)
    root = X.columns[t.tree_.feature[0]]
    print(f"seed {seed}: root split on '{root}' at {t.tree_.threshold[0]:.2f}")
'''),
md('''
A model whose top-level question changes with the sample is **high variance**. The fix is
to build many trees and average them - which is exactly what the next two models do.

## 2. Random Forest = bagging

**Bagging** (bootstrap aggregating):
1. Draw many random samples of the rows, with replacement.
2. Fit a deep tree on each one.
3. At every split, consider only a random subset of the features.
4. Average the votes.

Each tree overfits in its own direction; averaging cancels the noise out.
Trees are built **independently and in parallel**.
'''),
code('''
from sklearn.ensemble import RandomForestClassifier

for n in [1, 5, 25, 100, 300]:
    rf = RandomForestClassifier(n_estimators=n, random_state=42)
    s = cross_val_score(rf, X, y, cv=5)
    print(f"n_estimators={n:4d} | cv accuracy {s.mean():.3f} +/- {s.std():.3f}")
'''),
md('''
Notice that the score climbs and then plateaus. **More trees never hurt a Random Forest** -
they only cost time. That is unusual and worth remembering; you cannot overfit by adding
trees here (you can with boosting).

### Key hyperparameters

| Parameter | Effect |
|---|---|
| `n_estimators` | more trees, more stable, slower. 100-500 is normal |
| `max_depth` | `None` lets each tree grow fully - usually fine for a forest |
| `max_features` | features considered per split. `"sqrt"` is the classification default |
| `min_samples_leaf` | raise it to smooth the model |
| `class_weight` | `"balanced"` for imbalanced classes |

## 3. Gradient Boosting = boosting

**Boosting** is the opposite idea. Trees are built **sequentially**, and each new tree is
trained on the *errors of the ensemble so far*. Every tree is deliberately shallow (a
"weak learner"); the sum of hundreds of them is strong.

`HistGradientBoostingClassifier` is scikit-learn's fast modern implementation. It also
handles missing values natively, with no imputer.
'''),
code('''
from sklearn.ensemble import HistGradientBoostingClassifier

hgb = HistGradientBoostingClassifier(random_state=42)
s = cross_val_score(hgb, X, y, cv=5)
print(f"HistGradientBoosting | cv accuracy {s.mean():.3f} +/- {s.std():.3f}")
'''),
code('''
# learning_rate x n_iter is the central trade-off in boosting.
for lr in [0.01, 0.1, 0.5]:
    for n in [50, 200]:
        m = HistGradientBoostingClassifier(learning_rate=lr, max_iter=n, random_state=42)
        s = cross_val_score(m, X, y, cv=5)
        print(f"learning_rate={lr:<5} max_iter={n:<4} | cv {s.mean():.3f}")
'''),
md('''
> **The rule:** a smaller `learning_rate` needs more `max_iter`. Small steps, many of them,
> generalise better than few large steps - but cost more time. Unlike a Random Forest,
> **too many boosting rounds does overfit**, so use `early_stopping=True`.

## 4. Head-to-head
'''),
code('''
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

contenders = {
    "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
    "Decision Tree":       DecisionTreeClassifier(max_depth=5, random_state=42),
    "Random Forest":       RandomForestClassifier(n_estimators=200, random_state=42),
    "HistGradientBoosting": HistGradientBoostingClassifier(random_state=42),
}

rows = []
for name, m in contenders.items():
    s = cross_val_score(m, X, y, cv=5, scoring="roc_auc")
    rows.append({"model": name, "roc_auc": s.mean().round(3), "std": s.std().round(3)})

pd.DataFrame(rows).sort_values("roc_auc", ascending=False).reset_index(drop=True)
'''),
md('''
On a small, clean dataset like this one, logistic regression is often competitive.
**Always benchmark against a simple model.** If a boosted ensemble only beats logistic
regression by 0.005 AUC, ship the logistic regression - it is faster, interpretable and
easier to debug.

## 5. Feature importance, and its trap
'''),
code('''
rf = RandomForestClassifier(n_estimators=300, random_state=42).fit(X_train, y_train)

importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values()
importances.plot.barh(figsize=(8, 6), color="#4C78A8")
plt.title("Built-in (impurity) feature importance")
plt.xlabel("importance")
plt.tight_layout()
plt.show()
'''),
md('''
**The trap:** built-in importance is biased towards high-cardinality features - a column
with many distinct values gets more chances to look useful. It is computed on the
*training* data, so it also rewards features the model overfitted on.

**Permutation importance** is the honest version: shuffle one column in the *test* set and
measure how much the score drops. If nothing drops, the feature was not being used.
'''),
code('''
from sklearn.inspection import permutation_importance

perm = permutation_importance(rf, X_test, y_test, n_repeats=20, random_state=42, scoring="roc_auc")

comparison = pd.DataFrame({
    "built-in": rf.feature_importances_,
    "permutation": perm.importances_mean,
}, index=X.columns).sort_values("permutation", ascending=False).round(4)
comparison
'''),
md('''
Where the two columns disagree, trust permutation importance. And remember: importance
means *the model used it*, never *it causes the outcome*.

---
## Exercises

1. Plot a `max_depth=2` tree and write out, in plain sentences, the rules it learned.
   Would you be comfortable showing them to a domain expert?
2. Set `max_features=1` on the Random Forest. Score drops - explain why, in terms of what
   each individual tree can see.
3. Turn on `early_stopping=True` for `HistGradientBoostingClassifier` and print
   `.n_iter_`. How many rounds did it actually need?
4. Run the head-to-head comparison on the imbalanced dataset from notebook 05, scoring by
   `average_precision`. Does the ranking change?
5. **Stretch:** `pip install xgboost lightgbm` and add `XGBClassifier` and
   `LGBMClassifier` to the comparison. These are what Kaggle competitions actually run on.
'''),
])

# ---------------------------------------------------------------- 08
write(os.path.join(OUT, "08-hyperparameter-tuning.ipynb"), [
md('''
# 08 - Hyperparameter Tuning, Done Correctly

Notebook 01 called `RandomizedSearchCV` and reported the best score it found.
**That reported score is optimistically biased** - you searched hundreds of settings and
kept the luckiest. This notebook shows the correct procedure.

By the end you will be able to:
1. Distinguish parameters from hyperparameters.
2. Choose between grid search and randomised search.
3. Tune *inside* a pipeline, so preprocessing never leaks.
4. Keep an honest final estimate of performance.
'''),
code(HEART),
md('''
## 1. Parameters vs hyperparameters

- **Parameters** are learned by `.fit()` - the coefficients of a regression, the splits
  in a tree. You never set these.
- **Hyperparameters** are set by you *before* fitting - `max_depth`, `alpha`,
  `n_neighbors`, `learning_rate`. These are what you tune.

## 2. Set aside a final test set first

This split happens once, at the start, and the test set is not touched again until the
very last cell. Every tuning decision is made using cross-validation on the training set.
'''),
code('''
from sklearn.model_selection import train_test_split, StratifiedKFold

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

cv = StratifiedKFold(5, shuffle=True, random_state=42)
print(f"training set: {len(X_train)} rows   |   held-out test set: {len(X_test)} rows")
print("The test set is now locked away until the final cell.")
'''),
md('''
## 3. Grid search - exhaustive

`GridSearchCV` tries **every combination**. Reliable, but the cost explodes:
3 x 4 x 3 values with 5-fold CV is 36 x 5 = 180 model fits.
'''),
code('''
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

grid = {
    "n_estimators": [100, 300],
    "max_depth": [3, 5, 10, None],
    "min_samples_leaf": [1, 2, 4],
}

gs = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid=grid, cv=cv, scoring="roc_auc",
    n_jobs=-1, return_train_score=True)

gs.fit(X_train, y_train)

print(f"combinations tried : {len(gs.cv_results_['params'])}")
print(f"best CV roc_auc    : {gs.best_score_:.3f}")
print(f"best parameters    : {gs.best_params_}")
'''),
code('''
# Always look at the top few, not just the winner.
results = pd.DataFrame(gs.cv_results_)
cols = ["param_n_estimators", "param_max_depth", "param_min_samples_leaf",
        "mean_test_score", "std_test_score"]
results[cols].sort_values("mean_test_score", ascending=False).head(8).round(4)
'''),
md('''
> **Look at `std_test_score`.** If the top five settings are within one standard deviation
> of each other, they are tied. Pick the **simplest** one - a shallower tree, fewer
> estimators - not the one that happened to score 0.001 higher.

## 4. Randomised search - usually better value

`RandomizedSearchCV` samples `n_iter` random combinations from distributions. For the same
compute budget it usually finds a better setting, because most hyperparameters barely
matter and grid search wastes fits exploring them exhaustively.
'''),
code('''
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform

distributions = {
    "n_estimators": randint(100, 500),
    "max_depth": [3, 5, 10, 15, None],
    "min_samples_leaf": randint(1, 10),
    "max_features": uniform(0.1, 0.9),
}

rs = RandomizedSearchCV(
    RandomForestClassifier(random_state=42),
    param_distributions=distributions, n_iter=40,
    cv=cv, scoring="roc_auc", n_jobs=-1, random_state=42)

rs.fit(X_train, y_train)
print(f"fits used       : {40 * 5}  (grid search used {len(gs.cv_results_['params']) * 5})")
print(f"best CV roc_auc : {rs.best_score_:.3f}")
print(f"best parameters : {rs.best_params_}")
'''),
md('''
**Practical workflow:** randomised search wide first to find the promising region, then a
small grid search around the winner to refine. Do not start with a huge grid.

## 5. Tuning inside a pipeline

This is the part that matters most. When preprocessing is in the pipeline, it is refit on
each CV fold - so tuning cannot leak (notebook 04). Address a step's hyperparameter with
`stepname__paramname`.
'''),
code('''
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression

pipe = Pipeline([
    ("scale", StandardScaler()),
    ("select", SelectKBest(f_classif)),
    ("clf", LogisticRegression(max_iter=5000)),
])

pipe_grid = {
    "select__k": [5, 8, 10, 13],
    "clf__C": [0.01, 0.1, 1, 10, 100],
}

pipe_search = GridSearchCV(pipe, pipe_grid, cv=cv, scoring="roc_auc", n_jobs=-1)
pipe_search.fit(X_train, y_train)

print(f"best CV roc_auc : {pipe_search.best_score_:.3f}")
print(f"best parameters : {pipe_search.best_params_}")
'''),
md('''
Note `select__k` and `clf__C`: the double underscore reaches into a named step. You can
even search over the *model itself* by putting a list of estimators in the grid.

## 6. The honest final number

`best_score_` is biased upward - it is the maximum over many noisy estimates. The
unbiased number comes from the test set you locked away in section 2, used exactly once.
'''),
code('''
from sklearn.metrics import roc_auc_score, classification_report

best_model = pipe_search.best_estimator_          # already refit on all of X_train
test_auc = roc_auc_score(y_test, best_model.predict_proba(X_test)[:, 1])

print(f"CV score during tuning (optimistic) : {pipe_search.best_score_:.3f}")
print(f"held-out test score (honest)        : {test_auc:.3f}")
print()
print(classification_report(y_test, best_model.predict(X_test),
                            target_names=["no disease", "disease"], digits=3))
'''),
md('''
The honest number is usually lower. **That drop is normal** - report the held-out number,
not the tuning number.

### When you cannot spare a test set: nested CV

With very little data, a held-out set is too small to trust. **Nested cross-validation**
puts the whole search inside an outer CV loop. It is expensive but gives an unbiased
estimate of "how well does my *tuning procedure* generalise".
'''),
code('''
from sklearn.model_selection import cross_val_score

inner = StratifiedKFold(3, shuffle=True, random_state=1)
outer = StratifiedKFold(5, shuffle=True, random_state=2)

search = GridSearchCV(pipe, pipe_grid, cv=inner, scoring="roc_auc", n_jobs=-1)
nested = cross_val_score(search, X, y, cv=outer, scoring="roc_auc")

print(f"nested CV roc_auc: {nested.mean():.3f} +/- {nested.std():.3f}")
print("This is an unbiased estimate of the whole tune-and-fit procedure.")
'''),
md('''
## 7. Saving the finished model
'''),
code('''
import joblib
from pathlib import Path

Path("models").mkdir(exist_ok=True)
joblib.dump(best_model, "models/heart_disease_pipeline.joblib")

# The saved object contains the scaler, the selector AND the classifier -
# so it takes raw dataframes at predict time. That is the point of pipelines.
reloaded = joblib.load("models/heart_disease_pipeline.joblib")
print("reloaded test AUC:", round(roc_auc_score(y_test, reloaded.predict_proba(X_test)[:, 1]), 3))
print("\\nsample prediction on raw data:", reloaded.predict(X_test.head(3)))
'''),
md('''
---
## Exercises

1. Time both searches with `%%time`. How much slower is grid search, and did it find a
   better setting?
2. Add `"clf__class_weight": [None, "balanced"]` to `pipe_grid`. Does the tuner choose it?
   Now re-tune with `scoring="average_precision"` and see whether the answer changes.
3. Look at the top 10 rows of `pipe_search.cv_results_`. Are the scores meaningfully
   different, or all within one std? Which setting would you actually ship?
4. Run the same tuning with `scoring="recall"` instead of `"roc_auc"`. How do the chosen
   hyperparameters change, and what does that tell you about the choice of metric?
5. **Stretch:** search over two different classifiers in one go by putting
   `{"clf": [LogisticRegression(max_iter=5000), RandomForestClassifier()]}` in the grid
   as a list of dicts, each with its own parameters.
'''),
])
