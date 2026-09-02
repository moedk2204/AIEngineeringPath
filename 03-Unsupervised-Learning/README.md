# Module 03 — Unsupervised Learning

**Goal:** find structure in data that has no labels — and defend it, because there is no
accuracy score to hide behind.

There is no `y`. The consequence is bigger than it sounds: you cannot measure "correct".
Silhouette and inertia tell you whether the clusters are geometrically tidy, never whether
they are useful. That judgement is yours to argue for.

## Order

| # | Notebook | Focus |
|---|---|---|
| 00 | `00-from-supervised-to-unsupervised.ipynb` | The mental shift, k-means and PCA basics — **start here** |
| 01 | `01-kmeans-from-scratch.ipynb` | Implement k-means yourself, no sklearn |
| 02 | `02-clustering-pca-dbscan-hierarchical.ipynb` | PCA, DBSCAN, hierarchical clustering on country data |
| 03 | `03-collaborative-recommendation.ipynb` | Item-based collaborative filtering on MovieLens |
| 04 | `04-news-clustering-nlp.ipynb` | TF-IDF + clustering on text |
| 05 | `05-churn-eda.ipynb` | EDA on churn data — the bridge back to supervised |

**Do not skip 01.** Writing the assignment/update loop yourself is the difference between
knowing what k-means does and knowing how to call it.

## Solutions

Worked solutions for notebooks 01–04 are in `solutions/`. Attempt the notebook first —
reading a solution feels like learning and is not. Use them to check, not to start.

## The three things students get wrong

1. **Not scaling before clustering.** K-means measures distance. Same lesson as module 02,
   notebook 02, and it bites harder here because there is no accuracy drop to warn you.
2. **Treating the elbow plot as an answer.** It is a hint. Silhouette gives you a second
   opinion. The business gives you the actual answer.
3. **Never profiling the clusters.** If you cannot write a plausible name for each cluster
   from its mean feature values, you have produced a colourful scatter plot, not a
   segmentation. Always finish with a profile table.

## Self-check

- [ ] Explain why there is no train/test split here.
- [ ] Choose `k` using two different methods and reconcile them.
- [ ] Say what DBSCAN can do that k-means cannot (hint: shape, and noise points).
- [ ] Explain what PCA gains you and what it costs you.
- [ ] Profile a set of clusters and give each one a business name.

## Datasets

From `../data/`: `Mall_Customers.csv`, `Country-data.csv`, `movies.csv`, `ratings.csv`,
`links.csv`, `tags.csv`, `news_data.tsv`, `churn.csv`.
