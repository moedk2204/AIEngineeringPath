# Setup

## 1. Check your Python

You need Python 3.9 or newer.

```bash
python --version
```

On Windows, if you have several versions installed, list them with `py -0p` and pick one
explicitly, e.g. `py -3.12 -m venv .venv`.

## 2. Create a virtual environment

From the folder containing this file:

**Windows (PowerShell)**
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

If PowerShell blocks the activation script:
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

**Windows (Git Bash / cmd)**
```bash
python -m venv .venv
source .venv/Scripts/activate      # Git Bash
.venv\Scripts\activate.bat         # cmd
```

**macOS / Linux**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

You should see `(.venv)` at the start of your prompt.

## 3. Install the packages

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

This takes a few minutes the first time.

## 4. Register the kernel and launch

```bash
python -m ipykernel install --user --name ml-learning-path --display-name "ML Learning Path"
jupyter lab
```

**Launch Jupyter from this folder**, not from inside a module folder. Every notebook loads
data with a path like `../data/heart-disease.csv`, which only resolves correctly when the
notebook is opened from its own module directory within this project.

In the notebook, check the kernel in the top right says **ML Learning Path**. If it says
something else, use *Kernel → Change Kernel*.

## 5. Verify

Open `01-ML-Libraries/01-numpy-introduction.ipynb` and run the first cell. Then run this
in any notebook:

```python
import numpy, pandas, matplotlib, sklearn
print(numpy.__version__, pandas.__version__, matplotlib.__version__, sklearn.__version__)

import pandas as pd
pd.read_csv("../data/heart-disease.csv").shape     # expect (303, 14)
```

If both work, you are set up.

---

## Common problems

**`ModuleNotFoundError` even though pip said it installed**
The notebook is running a different Python than the one you installed into. Check the
kernel name in the top-right of Jupyter, and confirm with:
```python
import sys; print(sys.executable)
```
It should point inside your `.venv`.

**`FileNotFoundError: ../data/heart-disease.csv`**
You launched Jupyter from the wrong directory, or moved a notebook out of its module
folder. Check with:
```python
import os; print(os.getcwd())
```
You should be inside `01-ML-Libraries`, `02-Supervised-Learning` or `03-Unsupervised-Learning`.

**Plots do not appear**
Add `%matplotlib inline` at the top of the notebook. On recent JupyterLab this is rarely
needed.

**`ConvergenceWarning` from `LogisticRegression`**
Not an error — the solver hit its iteration limit. Raise it: `LogisticRegression(max_iter=5000)`.
The notebooks already do this where needed.

**Deprecation warnings from scikit-learn**
The notebooks target scikit-learn 1.3+ and were verified against 1.8. If you see warnings
about removed parameters, check your version with `sklearn.__version__` and update the
notebook — the warning text usually names the replacement.

**A cell runs for a very long time**
`GridSearchCV` in `02/08` fits hundreds of models. Add `n_jobs=-1` (already set in the
notebooks) to use all cores, or reduce the grid while experimenting.

---

## Working without a virtual environment

It will usually work, but you risk breaking other projects on your machine. If you must:

```bash
pip install --user -r requirements.txt
```

## Behind a corporate proxy

If `pip install` fails with an SSL certificate error, your network is intercepting HTTPS.
Ask IT for the corporate root certificate and point pip at it:

```bash
pip config set global.cert /path/to/corporate-root.crt
```

Do **not** use `--trusted-host` to bypass verification on a shared machine.
