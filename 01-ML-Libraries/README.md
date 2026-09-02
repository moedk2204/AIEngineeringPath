# Module 01 — ML Libraries

**Goal:** move data around confidently. Nothing here is machine learning yet, and that is
the point — every hour you spend fighting pandas later is an hour you are not thinking
about the model.

## Order

| # | Notebook | Focus |
|---|---|---|
| 01 | `01-numpy-introduction.ipynb` | arrays, dtypes, shapes, broadcasting, indexing, aggregation |
| 02 | `02-numpy-exercises.ipynb` | practice — try it without looking back at 01 |
| 03 | `03-pandas-exercises.ipynb` | Series/DataFrame, selection, `groupby`, missing data, `apply` |
| 04 | `04-matplotlib-exercises.ipynb` | figure/axes, subplots, styling, plotting from a DataFrame |

These are **exercise** notebooks: each cell states a task in a comment and you write the
code underneath. Do not read ahead for answers.

## What you must be able to do before module 02

- [ ] Create an array of a given shape and explain what `.shape` and `.dtype` mean.
- [ ] Explain broadcasting with an example.
- [ ] Load a CSV, inspect it with `.info()` / `.describe()`, and find missing values.
- [ ] Select rows by condition and columns by name in one expression.
- [ ] Group by a column and aggregate two others differently.
- [ ] Fill or drop missing values, and say why you chose one over the other.
- [ ] Draw a figure with two subplots and label both axes.

If any box is unticked, redo that notebook. Module 02 assumes all seven.

## Datasets

Loaded from `../data/`: `car-sales.csv`, `car-sales-missing-data.csv`, `heart-disease.csv`.

## Source

Practice notebooks from the [Zero to Mastery ML course](https://github.com/mrdbourke/zero-to-mastery-ml).
