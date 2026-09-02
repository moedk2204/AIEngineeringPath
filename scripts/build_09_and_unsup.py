import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _nbutil import md, code, write

BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
SUP = os.path.join(BASE, "02-Supervised-Learning")
UNS = os.path.join(BASE, "03-Unsupervised-Learning")

# ---------------------------------------------------------------- 09 capstone
write(os.path.join(SUP, "09-capstone-project.ipynb"), [
md('''
# 09 - Capstone: Heart Disease Classifier

No new concepts. This is where you prove you can run the whole loop yourself.

**The brief.** A clinic wants a tool that flags patients who may have heart disease so a
doctor reviews them sooner. A missed case is far worse than a false alarm - a false alarm
costs one extra consultation, a miss can cost a life.

**That last sentence is the whole specification.** It tells you which metric to optimise
and where to put the decision threshold. Re-read it before you choose anything.

---
## Deliverables

1. A single `Pipeline` that takes a raw dataframe and returns predictions.
2. An honest, cross-validated performance estimate.
3. A stated decision threshold, with the reason for it.
4. Three sentences a non-technical clinic manager could read.

Work through the checklist below. Write your code in the empty cells.
'''),
code('''
import pandas as pd, numpy as np
import matplotlib.pyplot as plt

pd.set_option("display.max_columns", 50)
np.random.seed(42)

heart = pd.read_csv("../data/heart-disease.csv")
heart.head()
'''),
md('''
## Step 1 - Understand the data before modelling

- [ ] How many rows and columns?
- [ ] Any missing values?
- [ ] Is the target balanced?
- [ ] Which features look numeric but are really categorical? (`cp`, `restecg`, `slope`,
      `thal` are candidates - check how many distinct values each has.)
- [ ] Plot the distribution of two or three features split by target.

**Data dictionary:** `age`, `sex` (1=male), `cp` chest pain type, `trestbps` resting blood
pressure, `chol` cholesterol, `fbs` fasting blood sugar > 120, `restecg` resting ECG,
`thalach` max heart rate, `exang` exercise-induced angina, `oldpeak` ST depression,
`slope`, `ca` number of major vessels, `thal` thalassemia, `target` (1 = disease).
'''),
code('''
# your exploration here
'''),
md('''
## Step 2 - Split before you touch anything

- [ ] Hold out a test set with `stratify=y`.
- [ ] Say out loud: this test set is not used again until Step 7.
'''),
code('''
# your split here
'''),
md('''
## Step 3 - Build a baseline

- [ ] Score a `DummyClassifier(strategy="most_frequent")`. Write the number down.
- [ ] Score a plain `LogisticRegression` inside a `StandardScaler` pipeline.

Every later model must beat both. If your fancy model does not beat the baseline,
the fancy model is wrong, not the baseline.
'''),
code('''
# your baseline here
'''),
md('''
## Step 4 - Preprocessing pipeline

- [ ] Scale the continuous columns.
- [ ] One-hot encode the genuinely categorical ones (`handle_unknown="ignore"`).
- [ ] Assemble with `ColumnTransformer`.
- [ ] Confirm no fitting happens outside the pipeline. (Notebook 04.)
'''),
code('''
# your preprocessing here
'''),
md('''
## Step 5 - Compare models honestly

- [ ] Cross-validate at least four: Logistic Regression, KNN, Random Forest,
      HistGradientBoosting.
- [ ] Use `StratifiedKFold(5, shuffle=True, random_state=42)` for all of them.
- [ ] Score with `roc_auc` **and** `recall`. Report mean and std.
- [ ] Produce a small comparison table sorted by your chosen metric.

**Think:** given the brief, which metric decides the winner?
'''),
code('''
# your model comparison here
'''),
md('''
## Step 6 - Tune the winner

- [ ] `RandomizedSearchCV` over a sensible distribution, inside the pipeline.
- [ ] Check `std_test_score` on the top results - if they are tied, take the simpler model.
- [ ] Print the chosen hyperparameters.
'''),
code('''
# your tuning here
'''),
md('''
## Step 7 - Choose the decision threshold

The default 0.5 is almost certainly wrong for this brief.

- [ ] Get `predict_proba` on the held-out test set.
- [ ] Plot precision and recall against the threshold.
- [ ] Pick the threshold that reaches **at least 90% recall**.
- [ ] Report the precision you accept in exchange, and translate it into plain words:
      "to catch 9 out of 10 real cases we will also send N healthy patients for review".
'''),
code('''
# your threshold analysis here
'''),
md('''
## Step 8 - Final report

- [ ] Confusion matrix at your chosen threshold.
- [ ] `classification_report`.
- [ ] Permutation importance - which features drive the model?
- [ ] Save the pipeline with `joblib`, reload it, and predict on three raw rows.
'''),
code('''
# your final evaluation here
'''),
md('''
## Step 9 - Write it up

Fill this in. If you cannot complete it, you do not understand your own model yet.

> **Model:** ...
> **Cross-validated ROC-AUC:** ... +/- ...
> **Held-out test ROC-AUC:** ...
> **Chosen threshold:** ... , giving recall ... and precision ...
> **In plain words:** out of 100 patients with heart disease, this tool flags about ... of
> them. It also flags about ... healthy patients, who get one extra consultation.
> **Biggest limitation:** ...

---
## Grading yourself

| | Done right |
|---|---|
| Split | test set held out before any fitting, stratified |
| Leakage | every transformer lives inside the pipeline |
| Baseline | dummy and logistic regression both scored and beaten |
| Metric | chosen from the brief, not from habit |
| Threshold | moved deliberately, with a stated recall target |
| Honesty | final number comes from the held-out set, used once |
| Communication | the plain-words paragraph is filled in |

## Going further

- Repeat the entire process on `../data/churn.csv` (predict `Churn`). Same steps,
  different domain - a false alarm now costs a retention discount, not a consultation.
- Then move on to **03-Unsupervised-Learning**, where there is no target column at all.
'''),
])

# ---------------------------------------------------------------- unsupervised bridge
write(os.path.join(UNS, "00-from-supervised-to-unsupervised.ipynb"), [
md('''
# 00 - From Supervised to Unsupervised

Everything in module 02 had a `y`. You knew the right answer for every training row, and
"good" meant matching it.

**Unsupervised learning has no `y`.** You have features only, and you are looking for
structure that was already in the data. Which means:

| | Supervised | Unsupervised |
|---|---|---|
| Input | `X` and `y` | `X` only |
| Goal | predict `y` for new rows | find structure in `X` |
| "Correct" | measurable against `y` | **there is no correct** |
| Evaluation | accuracy, ROC-AUC, RMSE | silhouette, inertia, and human judgement |
| Typical use | fraud detection, pricing, diagnosis | customer segments, recommendations, compression, anomaly detection |

That fourth row is the one that catches people. You cannot compute an accuracy. You have
to argue that the structure you found is useful - and the metrics below only help you
argue, they do not settle it.

This notebook gives you the two tools you will use everywhere: **k-means** (grouping rows)
and **PCA** (compressing columns). The notebooks that follow go much deeper.
'''),
code('''
import pandas as pd, numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

mall = pd.read_csv("../data/Mall_Customers.csv")
print(mall.shape)
mall.head()
'''),
md('''
## 1. Look before you cluster

There is no target column here. Just customers.
'''),
code('''
print(mall.dtypes)
print()
print(mall.describe().round(1))
'''),
code('''
features = ["Annual_Income_(k$)", "Spending_Score"]
X = mall[features]

plt.figure(figsize=(7, 5))
plt.scatter(X[features[0]], X[features[1]], s=35, color="#4C78A8")
plt.xlabel(features[0]); plt.ylabel(features[1])
plt.title("Can you see groups here? K-means will try to.")
plt.grid(alpha=.3)
plt.show()
'''),
md('''
## 2. K-means in four lines

The algorithm:
1. Drop `k` centre points at random.
2. Assign each row to its nearest centre.
3. Move each centre to the mean of its assigned rows.
4. Repeat 2-3 until nothing moves.

Notebook 01 makes you implement exactly this from scratch. Here is the library version.
'''),
code('''
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# K-means measures distance, so scaling is mandatory - same lesson as notebook 02 in module 02.
X_scaled = StandardScaler().fit_transform(X)

kmeans = KMeans(n_clusters=5, n_init=10, random_state=42)
labels = kmeans.fit_predict(X_scaled)

plt.figure(figsize=(7, 5))
plt.scatter(X[features[0]], X[features[1]], c=labels, cmap="viridis", s=40)
centres = StandardScaler().fit(X).inverse_transform(kmeans.cluster_centers_)
plt.scatter(centres[:, 0], centres[:, 1], marker="X", s=250, c="red", label="centroids")
plt.xlabel(features[0]); plt.ylabel(features[1])
plt.title("K-means with k=5")
plt.legend(); plt.grid(alpha=.3)
plt.show()
'''),
md('''
## 3. But how do you choose `k`?

This is the central question of clustering, and there is no clean answer. Two tools help.

**Elbow method** - plot inertia (total squared distance to the nearest centroid) against
`k`. Inertia always falls as `k` rises, so you look for the bend where it stops falling fast.

**Silhouette score** - for each point, how much closer is it to its own cluster than to
the next nearest? Ranges from -1 to 1. Higher is better, and unlike inertia it has a peak.
'''),
code('''
from sklearn.metrics import silhouette_score

ks = range(2, 11)
inertias, silhouettes = [], []

for k in ks:
    km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(X_scaled)
    inertias.append(km.inertia_)
    silhouettes.append(silhouette_score(X_scaled, km.labels_))

fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
axes[0].plot(ks, inertias, "o-", color="#4C78A8")
axes[0].set_xlabel("k"); axes[0].set_ylabel("inertia")
axes[0].set_title("Elbow method - look for the bend"); axes[0].grid(alpha=.3)

axes[1].plot(ks, silhouettes, "o-", color="#F58518")
best_k = ks[int(np.argmax(silhouettes))]
axes[1].axvline(best_k, ls="--", c="grey", label=f"best k = {best_k}")
axes[1].set_xlabel("k"); axes[1].set_ylabel("silhouette score")
axes[1].set_title("Silhouette - this one has a real peak")
axes[1].legend(); axes[1].grid(alpha=.3)
plt.tight_layout()
plt.show()
'''),
md('''
> **The honest position.** These metrics measure whether the clusters are geometrically
> tidy. They cannot tell you whether the clusters are *meaningful to the business*.
> A silhouette of 0.55 with five segments the marketing team recognises beats a silhouette
> of 0.71 with two segments nobody can act on. **Always profile your clusters.**

## 4. Profile the clusters - the step people skip
'''),
code('''
mall["cluster"] = labels

profile = mall.groupby("cluster")[["Age"] + features].mean()
profile["count"] = mall["cluster"].value_counts().sort_index()
profile.round(1)
'''),
md('''
Now name them. Something like "young high spenders, low income" or "wealthy but cautious".
If you cannot write a plausible name for a cluster, it probably is not a real segment.

## 5. PCA - compressing columns

K-means groups **rows**. PCA compresses **columns**: it finds new axes that capture as
much variance as possible, so you can plot 13 dimensions in 2.
'''),
code('''
from sklearn.decomposition import PCA

# Use the heart-disease features from module 02 - 13 columns, impossible to plot directly.
heart = pd.read_csv("../data/heart-disease.csv")
Xh = StandardScaler().fit_transform(heart.drop("target", axis=1))

pca = PCA()
pca.fit(Xh)

cumulative = np.cumsum(pca.explained_variance_ratio_)

plt.figure(figsize=(8, 4))
plt.bar(range(1, len(cumulative) + 1), pca.explained_variance_ratio_,
        color="#4C78A8", label="per component")
plt.plot(range(1, len(cumulative) + 1), cumulative, "o-", color="#E45756", label="cumulative")
plt.axhline(0.9, ls="--", c="grey", label="90% of variance")
plt.xlabel("principal component"); plt.ylabel("explained variance ratio")
plt.title("How many components do you actually need?")
plt.legend(); plt.grid(alpha=.3)
plt.show()

print(f"components needed for 90% of the variance: {np.argmax(cumulative >= 0.9) + 1} of {Xh.shape[1]}")
'''),
code('''
# Project 13 dimensions down to 2 and colour by the (unused) target, just to see.
coords = PCA(n_components=2).fit_transform(Xh)

plt.figure(figsize=(7, 5))
for value, colour, name in [(0, "#4C78A8", "no disease"), (1, "#E45756", "disease")]:
    mask = heart["target"] == value
    plt.scatter(coords[mask, 0], coords[mask, 1], s=30, c=colour, label=name, alpha=.7)
plt.xlabel("PC1"); plt.ylabel("PC2")
plt.title("PCA never saw the labels - yet the classes partly separate")
plt.legend(); plt.grid(alpha=.3)
plt.show()
'''),
md('''
**That is the point of PCA.** It found the directions of greatest variance using the
features alone, and those directions happen to line up with the thing we care about.
That is why PCA is useful as a preprocessing step: fewer columns, less noise, faster models.

**Warning:** PC1 and PC2 are mixtures of all original features. You gain a plot and lose
interpretability. Never PCA a dataset you need to explain to a regulator without a plan.

---
## Exercises

1. Re-run k-means on the mall data **without scaling**. Do the clusters change? Relate
   your answer to module 02, notebook 02.
2. Add `Age` as a third clustering feature. Does the best `k` by silhouette change?
   Profile the new clusters and try to name them.
3. Run k-means with `n_init=1` and several different `random_state` values. Do you get the
   same clusters every time? What does `n_init=10` protect you against?
4. Fit `PCA(n_components=5)` on the heart data, then train a logistic regression on those
   5 components. Compare accuracy against using all 13 features. What did you trade?
5. **Stretch:** cluster the heart-disease patients ignoring `target`, then cross-tabulate
   your clusters against the real `target`. Did unsupervised learning rediscover the disease?

---
## Next

- **01** - implement k-means yourself, from scratch. Do not skip it.
- **02** - PCA, DBSCAN and hierarchical clustering on real country data.
- **03** - collaborative filtering: recommendations from a ratings matrix.
- **04** - clustering text documents with TF-IDF.
- **05** - EDA on churn data, the bridge back to a supervised problem.
'''),
])
