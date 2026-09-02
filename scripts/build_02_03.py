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
heart.head()
'''

# ---------------------------------------------------------------- 02
write(os.path.join(OUT, "02-preprocessing-and-scaling.ipynb"), [
md('''
# 02 - Preprocessing & Feature Scaling

**Why this notebook exists:** in notebook 01 you called `KNeighborsClassifier` and `SVC`
without ever scaling the data. Those two models measure *distance* between points,
so a column measured in hundreds (`chol`) silently drowns out a column measured in 0/1 (`sex`).

By the end you will be able to:
1. Show, with numbers, that scaling changes KNN/SVM accuracy.
2. Pick between `StandardScaler`, `MinMaxScaler` and `RobustScaler`.
3. Encode categorical columns and fill missing values.
4. Wire it all into a single `ColumnTransformer`.

**Rule of thumb:** scale for KNN, SVM, Logistic Regression, PCA and neural nets.
Trees (Decision Tree, Random Forest, Gradient Boosting) do **not** need it.
'''),
code(HEAD),
md('''
## 1. The raw feature ranges

Look at how different the scales are. `chol` runs to roughly 560, `sex` is 0 or 1.
'''),
code('''
X.describe().T[["min", "max", "mean", "std"]].round(2)
'''),
md('''
## 2. KNN without scaling vs with scaling

Same model, same split, same seed. The only difference is one `StandardScaler`.
'''),
code('''
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

raw = KNeighborsClassifier().fit(X_train, y_train)
scaled = Pipeline([
    ("scaler", StandardScaler()),
    ("knn", KNeighborsClassifier()),
]).fit(X_train, y_train)

print(f"KNN, raw features    : {raw.score(X_test, y_test):.3f}")
print(f"KNN, scaled features : {scaled.score(X_test, y_test):.3f}")
'''),
md('''
> **Read the gap.** That difference is not a better algorithm - it is the same algorithm
> finally able to see all the features. This is the most common silent bug for beginners.

Now the same test for SVM.
'''),
code('''
from sklearn.svm import SVC

raw_svm = SVC().fit(X_train, y_train)
scaled_svm = Pipeline([("scaler", StandardScaler()), ("svm", SVC())]).fit(X_train, y_train)

print(f"SVM, raw features    : {raw_svm.score(X_test, y_test):.3f}")
print(f"SVM, scaled features : {scaled_svm.score(X_test, y_test):.3f}")
'''),
code('''
# Control group: a tree, which is scale-invariant by design.
from sklearn.ensemble import RandomForestClassifier

raw_rf = RandomForestClassifier(random_state=42).fit(X_train, y_train)
scaled_rf = Pipeline([("scaler", StandardScaler()),
                      ("rf", RandomForestClassifier(random_state=42))]).fit(X_train, y_train)

print(f"Random Forest, raw    : {raw_rf.score(X_test, y_test):.3f}")
print(f"Random Forest, scaled : {scaled_rf.score(X_test, y_test):.3f}   <- basically unchanged")
'''),
md('''
## 3. Which scaler?

| Scaler | What it does | Use when |
|---|---|---|
| `StandardScaler` | mean 0, std 1 | default choice, roughly bell-shaped data |
| `MinMaxScaler` | squeezes into [0, 1] | you need a bounded range (e.g. image pixels) |
| `RobustScaler` | centres on median, scales by IQR | the column has outliers |
'''),
code('''
from sklearn.preprocessing import MinMaxScaler, RobustScaler

col = X[["chol"]]
comparison = pd.DataFrame({
    "original":  col["chol"].values,
    "standard":  StandardScaler().fit_transform(col).ravel(),
    "minmax":    MinMaxScaler().fit_transform(col).ravel(),
    "robust":    RobustScaler().fit_transform(col).ravel(),
})
comparison.describe().T[["min", "max", "mean", "std"]].round(2)
'''),
code('''
fig, axes = plt.subplots(1, 4, figsize=(16, 3))
for ax, name in zip(axes, comparison.columns):
    ax.hist(comparison[name], bins=30, color="#4C78A8")
    ax.set_title(name)
plt.suptitle("Same column, four scalings - watch the x-axis, not the shape")
plt.tight_layout()
plt.show()
'''),
md('''
**Notice the shape never changes.** Scaling moves and stretches the axis; it does not
make a skewed column normal. That is a different tool (log transform, `PowerTransformer`).

## 4. Categorical columns and missing values

`heart-disease.csv` is all numeric. Switch to the car sales data, which has both text
columns and missing values - the realistic case.
'''),
code('''
cars = pd.read_csv("../data/car-sales-extended-missing-data.csv")
print(cars.dtypes)
print("\\nmissing values per column:")
print(cars.isna().sum())
cars.head()
'''),
code('''
# Rows with no target are useless for supervised learning - drop them. Never impute y.
cars = cars.dropna(subset=["Price"])

X_cars = cars.drop("Price", axis=1)
y_cars = cars["Price"]

categorical = ["Make", "Colour"]
numeric = ["Odometer (KM)", "Doors"]
print(f"{len(X_cars)} rows kept")
'''),
md('''
### The `ColumnTransformer`

Different column types need different treatment. `ColumnTransformer` applies a
separate mini-pipeline to each group and glues the results back together.
'''),
code('''
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor

categorical_pipe = Pipeline([
    ("impute", SimpleImputer(strategy="constant", fill_value="missing")),
    ("encode", OneHotEncoder(handle_unknown="ignore")),
])

numeric_pipe = Pipeline([
    ("impute", SimpleImputer(strategy="median")),
    ("scale", StandardScaler()),
])

preprocessor = ColumnTransformer([
    ("cat", categorical_pipe, categorical),
    ("num", numeric_pipe, numeric),
])

model = Pipeline([
    ("prep", preprocessor),
    ("rf", RandomForestRegressor(random_state=42)),
])

Xc_train, Xc_test, yc_train, yc_test = train_test_split(
    X_cars, y_cars, test_size=0.2, random_state=42)

model.fit(Xc_train, yc_train)
print(f"R^2 on test set: {model.score(Xc_test, yc_test):.3f}")
'''),
code('''
# What did the preprocessor actually produce?
transformed = preprocessor.fit_transform(Xc_train)
print(f"{Xc_train.shape[1]} original columns -> {transformed.shape[1]} numeric columns")
print(preprocessor.get_feature_names_out())
'''),
md('''
### Why `handle_unknown="ignore"` matters

If the test set contains a car colour that never appeared in training, a strict encoder
crashes at predict time. `"ignore"` encodes it as all zeros instead. In production that is
the difference between a working API and a 500 error.

---
## Exercises

1. Change the KNN comparison to use `MinMaxScaler` instead of `StandardScaler`.
   Does the conclusion change?
2. Set `n_neighbors=1` and re-run the raw-vs-scaled comparison. Does scaling matter
   *more* or *less* for a very flexible model? Explain why.
3. In the car sales pipeline, swap `SimpleImputer(strategy="median")` for `"mean"`.
   Look at the `Odometer (KM)` distribution first and predict which will do better.
4. Use `RobustScaler` in the numeric pipeline instead of `StandardScaler` and compare R^2.
5. **Stretch:** `Doors` is numeric but really categorical (3, 4, 5). Move it into the
   categorical group and see whether the score improves.
'''),
])

# ---------------------------------------------------------------- 03
write(os.path.join(OUT, "03-overfitting-and-learning-curves.ipynb"), [
md('''
# 03 - Overfitting, Underfitting & Learning Curves

Notebook 01 reported a single test score. A single number cannot tell you *why* a model
is bad - too simple, too complex, or just starved of data. This notebook gives you the
two diagnostic plots that answer that question.

By the end you will be able to:
1. Read the train-vs-test gap as a symptom.
2. Draw a **validation curve** to find the right model complexity.
3. Draw a **learning curve** to decide whether more data would help.
4. Say what bias and variance mean, in terms you can see on a plot.
'''),
code(HEAD),
md('''
## 1. The gap is the signal

Train a decision tree with no depth limit. It will memorise the training set.
'''),
code('''
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

for depth in [1, 3, 5, None]:
    tree = DecisionTreeClassifier(max_depth=depth, random_state=42).fit(X_train, y_train)
    tr, te = tree.score(X_train, y_train), tree.score(X_test, y_test)
    label = "unlimited" if depth is None else str(depth)
    print(f"max_depth={label:>9} | train {tr:.3f} | test {te:.3f} | gap {tr - te:+.3f}")
'''),
md('''
Three readings of that table:

- **`max_depth=1`**: both scores low. The model is too simple to capture the pattern -
  **underfitting**, also called **high bias**.
- **`max_depth=None`**: train score is a perfect 1.000, test score is much lower.
  The model memorised noise - **overfitting**, also called **high variance**.
- Somewhere in the middle is the sweet spot. Finding it by eye in a printed table is
  painful, which is what the next plot is for.

## 2. Validation curve - how complex should the model be?
'''),
code('''
from sklearn.model_selection import validation_curve

depths = np.arange(1, 21)
train_scores, val_scores = validation_curve(
    DecisionTreeClassifier(random_state=42), X, y,
    param_name="max_depth", param_range=depths,
    cv=5, scoring="accuracy")

train_mean, val_mean = train_scores.mean(axis=1), val_scores.mean(axis=1)

plt.figure(figsize=(9, 5))
plt.plot(depths, train_mean, "o-", label="training score", color="#4C78A8")
plt.plot(depths, val_mean, "o-", label="cross-validation score", color="#F58518")
plt.fill_between(depths, val_scores.min(axis=1), val_scores.max(axis=1),
                 alpha=.15, color="#F58518")
plt.axvline(depths[val_mean.argmax()], ls="--", c="grey",
            label=f"best depth = {depths[val_mean.argmax()]}")
plt.xlabel("max_depth  (model complexity ->)")
plt.ylabel("accuracy")
plt.title("Validation curve: the gap opens as the tree gets deeper")
plt.legend(); plt.grid(alpha=.3)
plt.show()
'''),
md('''
**How to read it.** The blue line climbs towards 1.0 - a deep enough tree always memorises
the training data. The orange line peaks and then flattens or falls. The distance between
the two lines *is* the variance. Pick the depth at the orange peak, not the blue one.

## 3. Learning curve - would more data help?

This is the question that decides whether you go collect more rows or go change the model.
'''),
code('''
from sklearn.model_selection import learning_curve
from sklearn.ensemble import RandomForestClassifier

def plot_learning_curve(estimator, X, y, title, ax):
    sizes, train_scores, val_scores = learning_curve(
        estimator, X, y, cv=5, scoring="accuracy",
        train_sizes=np.linspace(0.1, 1.0, 8))
    ax.plot(sizes, train_scores.mean(axis=1), "o-", label="training", color="#4C78A8")
    ax.plot(sizes, val_scores.mean(axis=1), "o-", label="validation", color="#F58518")
    ax.set_title(title)
    ax.set_xlabel("training examples"); ax.set_ylabel("accuracy")
    ax.grid(alpha=.3); ax.legend(); ax.set_ylim(0.5, 1.05)

fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
plot_learning_curve(DecisionTreeClassifier(max_depth=1, random_state=42), X, y,
                    "Underfit (max_depth=1)", axes[0])
plot_learning_curve(RandomForestClassifier(random_state=42), X, y,
                    "Overfit (unpruned forest)", axes[1])
plt.tight_layout()
plt.show()
'''),
md('''
**The decision rule:**

| What you see | Diagnosis | What to do |
|---|---|---|
| Both curves flat and **low**, close together | underfitting / high bias | more features, a more flexible model, less regularisation |
| Big **gap**, validation still rising | overfitting / high variance | **more data will help**, or regularise |
| Big gap, validation curve **flat** | overfitting, data-saturated | more data will *not* help - simplify or regularise instead |

That last row is the expensive mistake: paying to collect ten times more rows when the
curve had already flattened.

## 4. Seeing bias and variance directly

A regression example makes the geometry obvious.
'''),
code('''
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline

rng = np.random.RandomState(0)
xs = np.sort(rng.rand(30)) * 6
ys = np.sin(xs) + rng.randn(30) * 0.25
grid = np.linspace(0, 6, 300)

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, deg, title in zip(axes, [1, 4, 20],
                          ["Underfit (degree 1)", "Good fit (degree 4)", "Overfit (degree 20)"]):
    m = make_pipeline(PolynomialFeatures(deg), LinearRegression()).fit(xs.reshape(-1, 1), ys)
    ax.scatter(xs, ys, s=25, color="#333333", zorder=3, label="data")
    ax.plot(grid, m.predict(grid.reshape(-1, 1)), color="#E45756", lw=2, label="model")
    ax.plot(grid, np.sin(grid), "--", color="grey", lw=1, label="truth")
    ax.set_ylim(-2, 2); ax.set_title(title); ax.legend(fontsize=8)
plt.tight_layout()
plt.show()
'''),
md('''
Left: the model is a straight line, it cannot bend - **bias**.
Right: the model chases every noisy point and swings wildly between them - **variance**.
Middle: it tracks the grey truth line. That is what you are aiming for every time.

---
## Exercises

1. Redo the validation curve for `RandomForestClassifier` over `n_estimators`
   (`[10, 25, 50, 100, 200, 400]`). Does the gap widen the way it did for a single tree?
   Explain the difference.
2. Run a validation curve over `min_samples_leaf` for a decision tree
   (`[1, 2, 5, 10, 20, 50]`). Which direction on that axis means *more* complexity?
3. Plot a learning curve for a `max_depth=3` tree. Which of the three rows in the table
   above does it match, and what would you do next?
4. In the polynomial plot, change the noise from `0.25` to `0.6`. At what degree does
   overfitting become visible now? What does that say about noisy data?
5. **Stretch:** write `diagnose(estimator, X, y)` that fits a learning curve and prints
   "underfitting", "overfitting" or "well-fit" based on the final train/validation gap
   and level. Test it on all three models above.
'''),
])
