import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _nbutil import md, code, write

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "04-Deep-Learning")

# ---------------------------------------------------------------- 06
write(os.path.join(OUT, "06-pytorch-fundamentals.ipynb"), [
md('''
# 06 - PyTorch Fundamentals

Notebook 07 jumps straight into a convolutional network in PyTorch. This notebook is the
missing step before it: **tensors, autograd, `nn.Module`, and the training loop**.

Everything here maps onto notebook 01, where you wrote it all in NumPy by hand. The point
of this notebook is to see which of your hand-written pieces PyTorch replaces, and which it
still expects you to write.

| You wrote by hand (notebook 01) | PyTorch gives you |
|---|---|
| `forward()` | `nn.Module.forward` and layers like `nn.Linear` |
| `backward()` — the chain rule | `loss.backward()` (autograd) |
| `update()` | `optimizer.step()` |
| the epoch loop | **still yours to write** |
'''),
code('''
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

torch.manual_seed(42)
np.random.seed(42)

print("torch version:", torch.__version__)
device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device)
'''),
md('''
## 1. Tensors

A tensor is a NumPy array that can (a) live on a GPU and (b) remember how it was computed.
Everything else is familiar.
'''),
code('''
a = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
b = torch.ones(2, 2)

print("a =\\n", a)
print("\\na @ b =\\n", a @ b)
print("\\nshape", a.shape, "| dtype", a.dtype, "| device", a.device)

# Round-trip with NumPy
arr = np.array([1.0, 2.0, 3.0])
t = torch.from_numpy(arr)
print("\\nfrom numpy:", t, "-> back to numpy:", t.numpy())
'''),
md('''
### Three tensor gotchas that cost beginners hours

1. **dtype.** `torch.tensor([1, 2])` is `int64`; layers want `float32`. Use
   `torch.tensor(x, dtype=torch.float32)` or `.float()`.
2. **Shape.** `nn.Linear` expects `(batch_size, features)`. A 1-D tensor of shape `(n,)`
   is not the same as `(n, 1)`. Most "mat1 and mat2 shapes cannot be multiplied" errors
   are this.
3. **Device.** Model and data must be on the same device, or you get a runtime error.
'''),
code('''
x = torch.tensor([1, 2, 3])
print("int tensor:", x.dtype, "->", x.float().dtype)

v = torch.randn(5)
print("\\nshape (5,)   :", v.shape)
print("shape (5, 1) :", v.reshape(-1, 1).shape, " <- what a layer wants")
print("shape (1, 5) :", v.unsqueeze(0).shape)
'''),
md('''
## 2. Autograd — the part that replaces your `backward()`

Set `requires_grad=True` and PyTorch records every operation into a graph. Call
`.backward()` and it walks that graph backwards applying the chain rule — the same
derivation you did by hand in notebook 01.
'''),
code('''
# A trivial example you can verify with calculus.
x = torch.tensor(3.0, requires_grad=True)
y = x ** 2 + 5 * x + 1          # dy/dx = 2x + 5

y.backward()

print(f"y  = {y.item():.1f}")
print(f"dy/dx at x=3 = {x.grad.item():.1f}")
print(f"by hand: 2*3 + 5 = {2 * 3 + 5}")
'''),
code('''
# The same thing that mattered in notebook 01: gradients of a loss w.r.t. weights.
w = torch.randn(3, 1, requires_grad=True)
b = torch.zeros(1, requires_grad=True)
X = torch.randn(10, 3)
y_true = torch.randn(10, 1)

y_pred = X @ w + b
loss = ((y_pred - y_true) ** 2).mean()
loss.backward()

print("loss:", round(loss.item(), 4))
print("dL/dw shape:", w.grad.shape, " (matches w:", tuple(w.shape), ")")
print("dL/db:", b.grad)
'''),
md('''
> **The shape rule from notebook 01 still holds:** every gradient has the same shape as the
> parameter it belongs to. PyTorch enforces it for you now, but the reasoning is unchanged.

### Gradients accumulate — this is the classic bug

PyTorch **adds** new gradients to whatever is already in `.grad`. If you forget to clear
them, epoch 2 trains on the sum of epochs 1 and 2.
'''),
code('''
w = torch.tensor([1.0], requires_grad=True)

for step in range(3):
    loss = (w * 2).sum()
    loss.backward()
    print(f"step {step}: w.grad = {w.grad.item()}   <- should be 2.0 every time")

print("\\nNow with zeroing:")
w.grad = None
for step in range(3):
    loss = (w * 2).sum()
    loss.backward()
    print(f"step {step}: w.grad = {w.grad.item()}")
    w.grad = None                 # in real code: optimizer.zero_grad()
'''),
md('''
**That is why `optimizer.zero_grad()` is in every training loop you will ever read.**
Leave it out and your model trains on garbage while reporting no error at all.

## 3. Building a model with `nn.Module`

Notebook 01's `init_params` + `forward` become a class. `nn.Linear(in, out)` holds the
weight matrix and bias, and initialises them sensibly for you.
'''),
code('''
class SimpleNet(nn.Module):
    def __init__(self, n_input=2, n_hidden=16, n_output=1):
        super().__init__()
        self.fc1 = nn.Linear(n_input, n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_output)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        return self.fc2(x)          # raw logits - no sigmoid here, see the note below

model = SimpleNet()
print(model)
print("\\ntrainable parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))
'''),
md('''
### Why no sigmoid on the output

In notebook 01 the last step was `sigmoid`, and the loss was binary cross-entropy. In
PyTorch you return the raw **logits** and use `nn.BCEWithLogitsLoss`, which applies the
sigmoid internally in a numerically stable way. Applying sigmoid yourself *and* using
`BCEWithLogitsLoss` double-counts it and quietly ruins training.

Equivalent shorthand for simple stacks:
'''),
code('''
same_model = nn.Sequential(
    nn.Linear(2, 16),
    nn.ReLU(),
    nn.Linear(16, 1),
)
print(same_model)
'''),
md('''
## 4. Loss and optimizer

`optimizer.step()` replaces your hand-written `update()`. Adam is the sensible default —
it adapts the step size per parameter, so it is far less sensitive to the learning rate
than the plain gradient descent you wrote in notebook 01.
'''),
code('''
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

print("loss function:", criterion)
print("optimizer:", optimizer.__class__.__name__, "| lr =", optimizer.param_groups[0]["lr"])
'''),
md('''
## 5. The training loop

**Memorise this shape.** Every PyTorch project you will ever read has this loop in it.

```python
for epoch in range(epochs):
    optimizer.zero_grad()          # 1. clear old gradients
    output = model(X)              # 2. forward
    loss = criterion(output, y)    # 3. how wrong
    loss.backward()                # 4. backward (autograd)
    optimizer.step()               # 5. update weights
```

Five lines. Compare it to notebook 01 — it is the same three ideas, with the gradient
maths handed off to autograd.
'''),
code('''
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split

X_np, y_np = make_moons(n_samples=1000, noise=0.20, random_state=42)
X_tr, X_te, y_tr, y_te = train_test_split(X_np, y_np, test_size=0.25, random_state=42)

# NumPy -> tensors. Note float32 and the (n, 1) target shape.
X_train = torch.tensor(X_tr, dtype=torch.float32)
y_train = torch.tensor(y_tr, dtype=torch.float32).reshape(-1, 1)
X_test = torch.tensor(X_te, dtype=torch.float32)
y_test = torch.tensor(y_te, dtype=torch.float32).reshape(-1, 1)

print("X_train", X_train.shape, "| y_train", y_train.shape)
'''),
code('''
model = SimpleNet()
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

history = {"train_loss": [], "test_loss": []}

for epoch in range(500):
    model.train()
    optimizer.zero_grad()
    logits = model(X_train)
    loss = criterion(logits, y_train)
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        model.eval()
        with torch.no_grad():
            test_loss = criterion(model(X_test), y_test)
        history["train_loss"].append(loss.item())
        history["test_loss"].append(test_loss.item())

    if epoch % 100 == 0:
        print(f"epoch {epoch:4d} | train loss {loss.item():.4f}")

def accuracy(model, X, y):
    model.eval()
    with torch.no_grad():
        preds = (torch.sigmoid(model(X)) > 0.5).float()
    return (preds == y).float().mean().item()

print(f"\\ntrain accuracy: {accuracy(model, X_train, y_train):.3f}")
print(f"test accuracy : {accuracy(model, X_test, y_test):.3f}")
'''),
code('''
steps = np.arange(len(history["train_loss"])) * 10
plt.figure(figsize=(9, 4))
plt.plot(steps, history["train_loss"], label="train", color="#4C78A8")
plt.plot(steps, history["test_loss"], label="test", color="#F58518")
plt.xlabel("epoch"); plt.ylabel("BCE loss")
plt.title("Same curve as notebook 01 - different amount of code")
plt.legend(); plt.grid(alpha=.3)
plt.show()
'''),
md('''
### `model.train()` / `model.eval()` and `torch.no_grad()`

Two different things, both easy to forget:

- **`model.train()` vs `model.eval()`** switches the *behaviour* of dropout and batch norm.
  Dropout must be off at evaluation time. Forget `.eval()` and your test scores are noisy
  and pessimistic.
- **`torch.no_grad()`** stops autograd from building a graph. It saves memory and time,
  and has no effect on the numbers.

Use both whenever you evaluate.

## 6. Datasets and DataLoaders

Notebook 01 used the whole training set for every step. Real training uses **mini-batches** —
faster, and the noise in each batch actually helps generalisation.
'''),
code('''
from torch.utils.data import TensorDataset, DataLoader

train_ds = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)

print(f"{len(train_ds)} samples -> {len(train_loader)} batches of 32")

xb, yb = next(iter(train_loader))
print("one batch:", xb.shape, yb.shape)
'''),
code('''
# The training loop with batches - the version you will actually use.
model = SimpleNet()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

for epoch in range(30):
    model.train()
    epoch_loss = 0.0
    for xb, yb in train_loader:                  # <- the only new line
        optimizer.zero_grad()
        loss = criterion(model(xb), yb)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    if epoch % 10 == 0:
        print(f"epoch {epoch:3d} | mean batch loss {epoch_loss / len(train_loader):.4f}")

print(f"\\ntest accuracy: {accuracy(model, X_test, y_test):.3f}")
'''),
md('''
## 7. Saving and loading

Save the `state_dict` (the weights), not the whole model object — pickling a model class
breaks as soon as you refactor the code that defines it.
'''),
code('''
from pathlib import Path
Path("models").mkdir(exist_ok=True)

torch.save(model.state_dict(), "models/moons_net.pt")

reloaded = SimpleNet()                                    # rebuild the architecture
reloaded.load_state_dict(torch.load("models/moons_net.pt"))
reloaded.eval()

print("reloaded test accuracy:", round(accuracy(reloaded, X_test, y_test), 3))
'''),
md('''
---
## Exercises

1. Delete `optimizer.zero_grad()` from the training loop and run it. Describe what happens
   to the loss, and why no error is raised.
2. Add a sigmoid to the end of `SimpleNet.forward` while keeping `BCEWithLogitsLoss`.
   Does it still train? Explain what went wrong mathematically.
3. Swap `torch.optim.Adam` for `torch.optim.SGD(model.parameters(), lr=0.01)`. How many
   more epochs does SGD need to reach the same loss?
4. Add a third `nn.Linear` layer. Does test accuracy improve on this dataset? Connect your
   answer to notebook 00, rule 1.
5. Change `batch_size` to 1, then to 750 (the whole set). Time each and compare the loss
   curves. What are you trading?
6. **Stretch:** write your own `nn.Module` subclass for the *NumPy* network of notebook 01
   with no `nn.Linear` — use `nn.Parameter` and raw matrix multiplication. Confirm it
   trains identically.

## Next

Notebook 07 uses everything here on a convolutional network. The training loop will be the
same five lines.
'''),
])

# ---------------------------------------------------------------- 08
write(os.path.join(OUT, "08-training-diagnostics-and-regularization.ipynb"), [
md('''
# 08 - Training Diagnostics & Regularization

Module 02, notebook 03 taught you to read a learning curve for a scikit-learn model.
Neural networks need the same skill, plus a different toolkit for fixing what you find.

By the end you will be able to:
1. Read a train/validation loss curve and name the failure.
2. Diagnose a bad learning rate from the curve shape alone.
3. Apply dropout, weight decay and early stopping — and say what each one does.
4. Know which knob to reach for first.
'''),
code('''
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

torch.manual_seed(42)
np.random.seed(42)

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Deliberately hard: few samples, many features, plenty of noise. Overfitting is easy here.
X, y = make_classification(n_samples=600, n_features=40, n_informative=8,
                           n_redundant=5, flip_y=0.08, random_state=42)

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.4, random_state=42, stratify=y)
scaler = StandardScaler().fit(X_tr)

to_t = lambda a: torch.tensor(a, dtype=torch.float32)
X_train, X_test = to_t(scaler.transform(X_tr)), to_t(scaler.transform(X_te))
y_train, y_test = to_t(y_tr).reshape(-1, 1), to_t(y_te).reshape(-1, 1)

print("train", X_train.shape, "| validation", X_test.shape)
'''),
md('''
## 1. A reusable training function

Everything below changes one argument of this function and re-reads the curve.
'''),
code('''
def build(n_features=40, hidden=(256, 256), dropout=0.0, seed=42):
    torch.manual_seed(seed)          # seed BEFORE building, so every model starts identical
    layers, prev = [], n_features
    for h in hidden:
        layers += [nn.Linear(prev, h), nn.ReLU()]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        prev = h
    layers.append(nn.Linear(prev, 1))
    return nn.Sequential(*layers)

def train(model, epochs=300, lr=1e-3, weight_decay=0.0, patience=None):
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    hist = {"train": [], "val": []}
    best_val, best_epoch, best_state, waited = float("inf"), 0, None, 0

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        loss = criterion(model(X_train), y_train)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(X_test), y_test).item()

        hist["train"].append(loss.item())
        hist["val"].append(val_loss)

        if val_loss < best_val:
            best_val, best_epoch, waited = val_loss, epoch, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            waited += 1
            if patience is not None and waited >= patience:
                break

    hist["best_epoch"], hist["best_val"] = best_epoch, best_val
    hist["final_val"] = hist["val"][-1]
    hist["best_state"] = best_state
    return hist

def accuracy(model, X, y):
    model.eval()
    with torch.no_grad():
        return ((torch.sigmoid(model(X)) > 0.5).float() == y).float().mean().item()

def plot(hist, title, ax):
    ax.plot(hist["train"], label="train", color="#4C78A8")
    ax.plot(hist["val"], label="validation", color="#F58518")
    ax.axvline(hist["best_epoch"], ls="--", c="grey", lw=1,
               label=f"best epoch {hist['best_epoch']}")
    ax.set_xlabel("epoch"); ax.set_ylabel("BCE loss")
    ax.set_title(title, fontsize=10); ax.legend(fontsize=8); ax.grid(alpha=.3)
'''),
md('''
## 2. What overfitting looks like

Train a large network with no regularization at all.
'''),
code('''
baseline = build(hidden=(256, 256))
hist = train(baseline)

fig, ax = plt.subplots(figsize=(9, 4.5))
plot(hist, "No regularization", ax)
plt.show()

print(f"train accuracy      : {accuracy(baseline, X_train, y_train):.3f}")
print(f"validation accuracy : {accuracy(baseline, X_test, y_test):.3f}")
print(f"best validation loss was at epoch {hist['best_epoch']}, "
      f"but we trained for {len(hist['train'])}")
'''),
md('''
**The signature of overfitting in deep learning:** training loss keeps falling while
validation loss bottoms out and then turns *upward*. Everything after that turning point
is the network memorising noise.

Note the gap between the best epoch and the last epoch. Those wasted epochs actively made
the model worse — which is what the next four tools fix.

## 3. Learning rate — the first thing to check

Before reaching for regularization, check the learning rate. Its failures have distinctive
curve shapes.
'''),
code('''
fig, axes = plt.subplots(1, 3, figsize=(16, 4))
for ax, lr, label in zip(axes, [10.0, 1e-3, 1e-6],
                         ["Too high (10.0)", "About right (1e-3)", "Too low (1e-6)"]):
    h = train(build(hidden=(64, 64)), epochs=300, lr=lr)
    plot(h, label, ax)
plt.tight_layout()
plt.show()
'''),
md('''
| Curve shape | Diagnosis | Fix |
|---|---|---|
| Loss spikes, oscillates, or goes to `nan` | learning rate too **high** | divide by 10 |
| Loss falls smoothly then flattens with a rising validation line | about right, now overfitting | regularize |
| Loss barely moves, almost flat | learning rate too **low** | multiply by 10 |

Sweep by powers of ten (`1e-1, 1e-2, 1e-3, 1e-4`) — never in small increments.

## 4. Dropout

During training, dropout randomly zeroes a fraction of activations on every forward pass.
The network cannot rely on any single unit, so it is forced to spread the representation
out. At evaluation time it is switched off automatically by `model.eval()`.
'''),
code('''
fig, axes = plt.subplots(1, 3, figsize=(16, 4))
results = []
for ax, p in zip(axes, [0.0, 0.3, 0.6]):
    m = build(hidden=(256, 256), dropout=p)
    h = train(m)
    plot(h, f"dropout = {p}", ax)
    results.append({"dropout": p,
                    "best val loss": round(h["best_val"], 3),
                    "final val loss": round(h["final_val"], 3),
                    "val acc": round(accuracy(m, X_test, y_test), 3)})
plt.tight_layout()
plt.show()

import pandas as pd
pd.DataFrame(results)
'''),
md('''
**Read the two loss columns separately — they tell different stories.**

- **`best val loss` barely moves.** Dropout did not find a better model than no dropout.
- **`final val loss` at 0.6 is clearly lower.** Dropout slowed the runaway; the curve
  climbs less steeply after the minimum.

That is the honest result on this dataset, and it is more typical than tutorials suggest.
Dropout is a brake on overfitting, not an accuracy booster. If you are already stopping at
the best epoch, the brake matters much less.

Watch the *gap* between the two lines in the plots — that is what dropout is compressing.
And note that too much dropout eventually hurts the training loss too, which is
under-fitting. **0.2 to 0.5 is the normal range**; 0.6 is aggressive.

## 5. Weight decay (L2)

This is `Ridge` from module 02, notebook 06, in a neural network. It penalises large
weights, so the model prefers a smoother function. In PyTorch it is one argument on the
optimizer.
'''),
code('''
fig, axes = plt.subplots(1, 3, figsize=(16, 4))
results = []
for ax, wd in zip(axes, [0.0, 1e-3, 1e-1]):
    m = build(hidden=(256, 256))
    h = train(m, weight_decay=wd)
    plot(h, f"weight_decay = {wd}", ax)
    results.append({"weight_decay": wd,
                    "val acc": round(accuracy(m, X_test, y_test), 3),
                    "best val loss": round(h["best_val"], 4)})
plt.tight_layout()
plt.show()

pd.DataFrame(results)
'''),
md('''
A small, consistent gain at `1e-3`; clear damage at `1e-1`. That is the usual shape —
useful values live in a narrow band around `1e-4` to `1e-3`, and the improvement is real
but small. Sweep it on a log scale, never linearly.

## 6. Early stopping — the one that actually matters here

The cheapest regularizer of all: stop when validation loss stops improving, and restore
the weights from the best epoch. It costs nothing and needs no tuning beyond `patience`.
'''),
code('''
m = build(hidden=(256, 256))
h = train(m, epochs=300, patience=25)

print(f"stopped at epoch {len(h['train'])} of a possible 300")
print(f"best epoch was {h['best_epoch']}")

# Restore the best weights - without this, early stopping keeps the *worse* final weights.
m.load_state_dict(h["best_state"])
print(f"\\nvalidation accuracy at best epoch: {accuracy(m, X_test, y_test):.3f}")

fig, ax = plt.subplots(figsize=(9, 4.5))
plot(h, "Early stopping with patience=25", ax)
plt.show()
'''),
md('''
**Compare the numbers.** The unregularized baseline ended around a validation loss of 1.4;
its best epoch was around 0.48. Early stopping alone recovers that entire difference — far
more than dropout and weight decay achieved between them.

That is the real lesson of this notebook. The glamorous knobs are worth a few percent.
Stopping at the right epoch is worth three times as much, and it is free.

> **The trap:** early stopping without restoring the best weights gives you the weights from
> `patience` epochs *after* the optimum — the ones that triggered the stop. Always keep and
> reload the best state, as above.

## 7. Everything together
'''),
code('''
final = build(hidden=(256, 256), dropout=0.3)
h = train(final, epochs=400, lr=1e-3, weight_decay=1e-4, patience=40)
final.load_state_dict(h["best_state"])

fig, ax = plt.subplots(figsize=(9, 4.5))
plot(h, "dropout 0.3 + weight decay 1e-4 + early stopping", ax)
plt.show()

print(f"train accuracy      : {accuracy(final, X_train, y_train):.3f}")
print(f"validation accuracy : {accuracy(final, X_test, y_test):.3f}")
print(f"\\nbaseline validation accuracy was {accuracy(baseline, X_test, y_test):.3f}")
'''),
md('''
## The order to reach for these

1. **Check the learning rate.** Free, and it is the problem more often than you expect.
2. **Add early stopping.** Free, no tuning, never hurts.
3. **Get more data, or augment it.** Beats every other item on this list.
4. **Add dropout** (0.2–0.5) or **weight decay** (1e-5 to 1e-3).
5. **Make the network smaller.** A model that cannot memorise the data cannot overfit it.
6. Only then reach for exotic architectures.

Beginners tend to start at 6. Start at 1.

### One honest caveat

On a dataset this small, the differences in `best val loss` are within noise from run to
run. Change the `seed=` argument of `build()` and that ranking may shuffle (the `final val
loss` ordering is more stable). That is not a flaw in the
notebook — it is the lesson from module 02, notebook 04, reappearing: **a single run is a
lottery ticket.** Before claiming a regularizer helped, run it with three seeds.

---
## Exercises

1. Run the dropout comparison with `build(..., seed=0)`, `seed=1`, `seed=2`. Does the
   ranking of 0.0 / 0.3 / 0.6 hold on `best val loss`? On `final val loss`? What does that
   tell you about single-run comparisons?
2. Set `hidden=(16, 16)` with no regularization. Does it still overfit? Relate the answer
   to item 5 in the list above.
3. Sweep `weight_decay` over `[0, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]` and plot best validation
   loss against it on a log x-axis. Where is the minimum?
4. Add `torch.optim.lr_scheduler.ReduceLROnPlateau` to the training loop, halving the
   learning rate when validation loss stalls. Does it beat a fixed rate?
5. Replace the full-batch loop with a `DataLoader` at `batch_size=32` (notebook 06,
   section 6). Does mini-batch noise act as a regularizer here?
6. **Stretch:** add `nn.BatchNorm1d` after each linear layer. Explain why batch norm
   behaves differently in `train()` and `eval()` mode, and what breaks if you forget.
'''),
])
