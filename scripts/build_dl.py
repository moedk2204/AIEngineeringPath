import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _nbutil import md, code, write

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "04-Deep-Learning")

# ---------------------------------------------------------------- 00
write(os.path.join(OUT, "00-when-to-use-deep-learning.ipynb"), [
md('''
# 00 - When To Use Deep Learning (And When Not To)

Before any framework, one question: **does this problem need a neural network at all?**

The honest answer for most business data is no. Deep learning wins decisively on data with
*structure a human can perceive* — images, audio, text — and loses to gradient boosting on
spreadsheet-shaped data more often than beginners expect.

This notebook makes you prove that to yourself with three experiments. It uses only
scikit-learn and SciPy, so it runs before you install TensorFlow or PyTorch.
'''),
code('''
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
np.random.seed(42)

from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

cv = StratifiedKFold(5, shuffle=True, random_state=42)
'''),
md('''
## Experiment 1 — tabular data (303 rows, 13 columns)

The heart-disease dataset from module 02. This is what most real business data looks like.
'''),
code('''
heart = pd.read_csv("../data/heart-disease.csv")
X_tab, y_tab = heart.drop("target", axis=1), heart["target"]

contenders = {
    "Random Forest":        RandomForestClassifier(n_estimators=300, random_state=42),
    "HistGradientBoosting": HistGradientBoostingClassifier(random_state=42),
    "Neural net (MLP)":     make_pipeline(StandardScaler(),
                                MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=2000,
                                              random_state=42)),
}

rows = []
for name, model in contenders.items():
    s = cross_val_score(model, X_tab, y_tab, cv=cv, scoring="roc_auc")
    rows.append({"model": name, "roc_auc": round(s.mean(), 3), "std": round(s.std(), 3)})

pd.DataFrame(rows).sort_values("roc_auc", ascending=False).reset_index(drop=True)
'''),
md('''
Look at the ranking, and at the standard deviations. On 303 rows of tabular data, the
neural network has no advantage — and it needed scaling, more compute, and more
hyperparameters to get there.

**Why.** A neural net has to *learn* that features interact. A tree splits on interactions
directly. With only a few hundred rows there is nowhere near enough signal for the network
to discover what the tree gets for free.

## Experiment 2 — image data (1797 images, 64 pixels each)

Now the same three models on handwritten digits, where each row is an 8x8 image.
'''),
code('''
from sklearn.datasets import load_digits

digits = load_digits()
X_img, y_img = digits.data, digits.target
print(f"{X_img.shape[0]} images, {X_img.shape[1]} pixels each, {len(np.unique(y_img))} classes")

fig, axes = plt.subplots(1, 8, figsize=(12, 2))
for ax, img, label in zip(axes, digits.images, digits.target):
    ax.imshow(img, cmap="gray_r"); ax.set_title(label); ax.axis("off")
plt.suptitle("The raw data: 8x8 grayscale digits")
plt.show()
'''),
code('''
rows = []
for name, model in contenders.items():
    s = cross_val_score(model, X_img, y_img, cv=cv, scoring="accuracy")
    rows.append({"model": name, "accuracy": round(s.mean(), 3), "std": round(s.std(), 3)})

pd.DataFrame(rows).sort_values("accuracy", ascending=False).reset_index(drop=True)
'''),
md('''
**Read that carefully — and notice what did *not* happen.** All three are within about
half a point of each other. The neural network did not win.

This is worth dwelling on, because a lot of tutorials would quietly pick a friendlier
dataset here. The truth is that a plain fully-connected network on raw pixels has **no
built-in advantage on images**. It treats pixel 27 as just another column, exactly like the
tree does. 1797 tiny 8x8 images is nowhere near enough data for it to discover structure
on its own.

So what *does* make deep learning win on images? Not depth. **Architecture that already
knows something about images.** The next experiment shows what the raw-pixel models are
missing.

## Experiment 3 — the assumption every flat model makes

Train on the digits as they come. Then test on the *same* digits shifted one pixel down
and one pixel right. A human reads a shifted 3 as a 3 without noticing. Watch what the
models do.
'''),
code('''
from scipy.ndimage import shift as nd_shift
from sklearn.model_selection import train_test_split

images = digits.images / 16.0                       # scale pixels to 0-1
X_tr_img, X_te_img, y_tr_img, y_te_img = train_test_split(
    images, y_img, test_size=0.3, random_state=42, stratify=y_img)

def shift_all(batch, dy, dx):
    return np.stack([nd_shift(im, (dy, dx), order=1, mode="constant") for im in batch])

X_te_shifted = shift_all(X_te_img, 1, 1)            # one pixel down, one pixel right

flat = lambda a: a.reshape(len(a), -1)

rows = []
for name, model in [("Random Forest", RandomForestClassifier(n_estimators=300, random_state=42)),
                    ("Neural net (MLP)", MLPClassifier((128, 64), max_iter=3000, random_state=42))]:
    model.fit(flat(X_tr_img), y_tr_img)
    rows.append({
        "model": name,
        "centred digits": round(model.score(flat(X_te_img), y_te_img), 3),
        "shifted 1 pixel": round(model.score(flat(X_te_shifted), y_te_img), 3),
    })

pd.DataFrame(rows)
'''),
code('''
fig, axes = plt.subplots(2, 6, figsize=(11, 4))
for i in range(6):
    axes[0, i].imshow(X_te_img[i], cmap="gray_r"); axes[0, i].axis("off")
    axes[1, i].imshow(X_te_shifted[i], cmap="gray_r"); axes[1, i].axis("off")
axes[0, 0].set_title("original", loc="left", fontsize=9)
axes[1, 0].set_title("shifted 1px", loc="left", fontsize=9)
plt.suptitle("Obviously the same digits - to a human")
plt.show()
'''),
md('''
**That collapse is the whole argument for deep learning on images.**

Both models go from near-perfect to worse than useless, from a shift a human would not even
notice. Neither ever learned "a 3 is a shape". They learned "pixel 27 is dark and pixel 35
is light" — a lookup table keyed on absolute position. Move the digit and the table is wrong.

A **convolutional** network cannot make that mistake, because it is built differently: it
slides the same small filter across every position, so a stroke detected in the corner is
the same feature as a stroke detected in the middle. The invariance is baked into the
architecture, not learned from data.

That is the real answer to "why deep learning for images": not more layers, but **an
architecture whose assumptions match the data**. Notebooks 05 and 07 build exactly that.

## The decision table

| Your data | Start with | Reach for deep learning when |
|---|---|---|
| Spreadsheet / tabular | **Gradient boosting** (XGBoost, LightGBM, HistGB) | almost never — and only with a lot of rows |
| Images | CNN, or a pretrained model | immediately |
| Text | a transformer (module 05) | immediately |
| Audio / video | a neural network | immediately |
| Time series | start with gradient boosting on lag features | long sequences, many related series |

### Three practical rules

1. **Rows matter more than layers.** Deep learning is data-hungry. Under ~10,000 rows on
   tabular data, gradient boosting almost always wins.
2. **Never skip the baseline.** If your network cannot beat logistic regression, the
   network is wrong — not the baseline.
3. **You will rarely train from scratch.** In practice you fine-tune a pretrained model
   (notebook 09, and module 05). Training from zero is for learning, not for shipping.

### So why learn deep learning at all?

Because images, text and audio are where the interesting problems are, and because every
LLM in module 05 is a neural network. You need the mechanics before the abstractions make
sense. That is what notebook 01 is for.

---
## Exercises

1. Increase `hidden_layer_sizes` to `(256, 128, 64)` on the tabular experiment. Does the
   MLP catch up? What does that tell you about "just add more layers"?
2. Cut the digits dataset to the first 200 samples and re-run experiment 2. Which model
   degrades most? Relate the answer to rule 1 above.
3. Time all three models on the digits data with `%%time`. Compare accuracy gained against
   time spent — would you ship the winner?
4. Add `LogisticRegression` as a fourth contender in both experiments. How close does the
   simplest possible model get?
5. **Stretch:** flatten the digit images in a scrambled but *consistent* pixel order and
   re-run. The MLP score barely moves; a CNN's would collapse. Explain why — and what that
   says about what a CNN actually assumes about its input.
'''),
])

# ---------------------------------------------------------------- 01
write(os.path.join(OUT, "01-neural-net-from-scratch-numpy.ipynb"), [
md('''
# 01 - A Neural Network From Scratch, in NumPy

The single most important notebook in this module. You will build a working two-layer
network with **no framework** — forward pass, loss, backpropagation and gradient descent,
all in NumPy.

Once you have written backprop by hand, `model.fit()` stops being magic and starts being
a convenience. Everything in TensorFlow and PyTorch is this, optimised.

By the end you will be able to:
1. Compute a forward pass by hand.
2. Explain what a loss function measures.
3. Derive and implement backpropagation with the chain rule.
4. Train with gradient descent and watch the loss fall.
'''),
code('''
import numpy as np
import matplotlib.pyplot as plt
np.random.seed(42)

from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split

# Two interleaving half-circles - impossible for a straight line, easy for a small network.
X, y = make_moons(n_samples=1000, noise=0.20, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

plt.figure(figsize=(6, 5))
plt.scatter(X[:, 0], X[:, 1], c=y, cmap="coolwarm", s=18, edgecolor="k", linewidth=.3)
plt.title("make_moons - no straight line separates these")
plt.xlabel("x1"); plt.ylabel("x2")
plt.show()
'''),
md('''
## 1. One neuron

A neuron does two things:

$$z = \\mathbf{w} \\cdot \\mathbf{x} + b \\qquad a = \\sigma(z)$$

A weighted sum, then a non-linear **activation**. Without the activation, stacking layers
is pointless — a chain of linear maps is still just a linear map.
'''),
code('''
def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))   # clip stops overflow warnings

def relu(z):
    return np.maximum(0, z)

zs = np.linspace(-6, 6, 200)
fig, axes = plt.subplots(1, 2, figsize=(11, 3.5))
axes[0].plot(zs, sigmoid(zs), color="#4C78A8", lw=2); axes[0].set_title("sigmoid - squashes to (0, 1)")
axes[1].plot(zs, relu(zs), color="#F58518", lw=2);    axes[1].set_title("ReLU - max(0, z)")
for ax in axes:
    ax.grid(alpha=.3); ax.axhline(0, c="grey", lw=.5); ax.axvline(0, c="grey", lw=.5)
plt.show()
'''),
md('''
**Why ReLU is the default hidden activation.** Look at the sigmoid: for `z` beyond about
±5 the curve is flat, so its slope is nearly zero. During backprop those near-zero slopes
multiply together and the gradient vanishes before it reaches the early layers. ReLU has a
slope of exactly 1 wherever it is active, so the signal survives depth.

Sigmoid still earns its place on the **output** of a binary classifier, because there you
genuinely want a number between 0 and 1.

## 2. The architecture

Two inputs → a hidden layer of 16 ReLU units → one sigmoid output.
'''),
code('''
def init_params(n_input=2, n_hidden=16, n_output=1, seed=42):
    rng = np.random.RandomState(seed)
    # He initialisation: scale by sqrt(2/fan_in). Wrong scale here is a real bug -
    # too large and activations explode, too small and the signal dies out.
    return {
        "W1": rng.randn(n_input, n_hidden) * np.sqrt(2 / n_input),
        "b1": np.zeros((1, n_hidden)),
        "W2": rng.randn(n_hidden, n_output) * np.sqrt(2 / n_hidden),
        "b2": np.zeros((1, n_output)),
    }

params = init_params()
for k, v in params.items():
    print(f"{k}: shape {v.shape}")
print(f"\\ntotal trainable parameters: {sum(v.size for v in params.values())}")
'''),
md('''
## 3. Forward pass

$$Z_1 = XW_1 + b_1 \\quad A_1 = \\text{ReLU}(Z_1) \\quad Z_2 = A_1W_2 + b_2 \\quad \\hat{y} = \\sigma(Z_2)$$

We keep every intermediate value in a `cache` — backprop needs them.
'''),
code('''
def forward(X, params):
    Z1 = X @ params["W1"] + params["b1"]
    A1 = relu(Z1)
    Z2 = A1 @ params["W2"] + params["b2"]
    A2 = sigmoid(Z2)
    return A2, {"X": X, "Z1": Z1, "A1": A1, "Z2": Z2, "A2": A2}

preds, cache = forward(X_train, params)
print("output shape:", preds.shape)
print("first 5 predictions (untrained):", preds[:5].ravel().round(3))
'''),
md('''
Untrained, the outputs hover near 0.5 — the network knows nothing. Good.

## 4. The loss

**Binary cross-entropy** measures how wrong a probability is:

$$L = -\\frac{1}{m}\\sum \\left[ y\\log(\\hat{y}) + (1-y)\\log(1-\\hat{y}) \\right]$$

Predicting 0.99 when the answer is 1 costs almost nothing. Predicting 0.01 when the answer
is 1 costs a lot — the log blows up. That asymmetry is what makes it a good training signal.
'''),
code('''
def binary_cross_entropy(y_true, y_pred, eps=1e-9):
    y_true = y_true.reshape(-1, 1)
    y_pred = np.clip(y_pred, eps, 1 - eps)          # never log(0)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

print(f"loss before training: {binary_cross_entropy(y_train, preds):.4f}")
print(f"(a model guessing 0.5 every time scores {-np.log(0.5):.4f})")
'''),
md('''
## 5. Backpropagation

This is the part everyone finds hard, and it is just the **chain rule** applied backwards.

We want $\\partial L / \\partial W$ for every weight. Work from the output back:

| Step | Derivative |
|---|---|
| output layer error | $dZ_2 = \\hat{y} - y$ |
| output weights | $dW_2 = \\frac{1}{m} A_1^T dZ_2$ |
| propagate back | $dA_1 = dZ_2 W_2^T$ |
| through ReLU | $dZ_1 = dA_1 \\odot \\mathbb{1}[Z_1 > 0]$ |
| input weights | $dW_1 = \\frac{1}{m} X^T dZ_1$ |

The first line looks suspiciously simple. It is: sigmoid combined with cross-entropy makes
the derivative collapse to `prediction - truth`. That elegance is exactly why the pair is
used together.
'''),
code('''
def backward(y_true, cache, params):
    m = y_true.shape[0]
    y_true = y_true.reshape(-1, 1)

    dZ2 = cache["A2"] - y_true                       # sigmoid + BCE collapse to this
    dW2 = (cache["A1"].T @ dZ2) / m
    db2 = np.sum(dZ2, axis=0, keepdims=True) / m

    dA1 = dZ2 @ params["W2"].T
    dZ1 = dA1 * (cache["Z1"] > 0)                    # derivative of ReLU is 1 or 0
    dW1 = (cache["X"].T @ dZ1) / m
    db1 = np.sum(dZ1, axis=0, keepdims=True) / m

    return {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2}

grads = backward(y_train, cache, params)
for k, v in grads.items():
    print(f"d{k}: shape {v.shape}  (matches {k}: {params[k].shape})")
'''),
md('''
> **Debugging tip.** Every gradient must have exactly the same shape as the parameter it
> updates. If a shape mismatches, your backprop is wrong. Check this before anything else.

## 6. Gradient descent

Each gradient says "increasing this weight increases the loss by this much". So move the
opposite way, by a small step called the **learning rate**:

$$W \\leftarrow W - \\eta \\cdot dW$$
'''),
code('''
def update(params, grads, lr):
    for key in params:
        params[key] -= lr * grads[key]
    return params

def accuracy(X, y, params):
    probs, _ = forward(X, params)
    return ((probs.ravel() > 0.5).astype(int) == y).mean()

def train(X_tr, y_tr, X_te, y_te, epochs=2000, lr=0.5, n_hidden=16, verbose=True):
    params = init_params(n_hidden=n_hidden)
    history = {"train_loss": [], "test_loss": [], "train_acc": [], "test_acc": []}

    for epoch in range(epochs):
        preds, cache = forward(X_tr, params)          # forward
        grads = backward(y_tr, cache, params)         # backward
        params = update(params, grads, lr)            # step

        if epoch % 10 == 0:
            test_preds, _ = forward(X_te, params)
            history["train_loss"].append(binary_cross_entropy(y_tr, preds))
            history["test_loss"].append(binary_cross_entropy(y_te, test_preds))
            history["train_acc"].append(accuracy(X_tr, y_tr, params))
            history["test_acc"].append(accuracy(X_te, y_te, params))

        if verbose and epoch % 400 == 0:
            print(f"epoch {epoch:5d} | loss {binary_cross_entropy(y_tr, preds):.4f} "
                  f"| train acc {accuracy(X_tr, y_tr, params):.3f}")

    return params, history

trained, history = train(X_train, y_train, X_test, y_test)
print(f"\\nfinal train accuracy: {accuracy(X_train, y_train, trained):.3f}")
print(f"final test accuracy : {accuracy(X_test, y_test, trained):.3f}")
'''),
md('''
**That is the entire algorithm.** Three lines in the loop: forward, backward, update.
Every deep learning framework is an optimised, GPU-accelerated version of those three lines.
'''),
code('''
steps = np.arange(len(history["train_loss"])) * 10

fig, axes = plt.subplots(1, 2, figsize=(13, 4))
axes[0].plot(steps, history["train_loss"], label="train", color="#4C78A8")
axes[0].plot(steps, history["test_loss"], label="test", color="#F58518")
axes[0].set_xlabel("epoch"); axes[0].set_ylabel("cross-entropy loss")
axes[0].set_title("Loss - this is what 'learning' looks like")

axes[1].plot(steps, history["train_acc"], label="train", color="#4C78A8")
axes[1].plot(steps, history["test_acc"], label="test", color="#F58518")
axes[1].set_xlabel("epoch"); axes[1].set_ylabel("accuracy")
axes[1].set_title("Accuracy")
for ax in axes:
    ax.legend(); ax.grid(alpha=.3)
plt.tight_layout()
plt.show()
'''),
code('''
# What did it actually learn? Plot the decision boundary.
def plot_boundary(params, X, y, ax, title):
    pad = 0.5
    xx, yy = np.meshgrid(
        np.linspace(X[:, 0].min() - pad, X[:, 0].max() + pad, 300),
        np.linspace(X[:, 1].min() - pad, X[:, 1].max() + pad, 300))
    probs, _ = forward(np.c_[xx.ravel(), yy.ravel()], params)
    ax.contourf(xx, yy, probs.reshape(xx.shape), levels=25, cmap="coolwarm", alpha=.7)
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap="coolwarm", s=16, edgecolor="k", linewidth=.3)
    ax.set_title(title)

fig, ax = plt.subplots(figsize=(7, 5.5))
plot_boundary(trained, X_test, y_test, ax, "Learned decision boundary - curved, not linear")
plt.show()
'''),
md('''
## 7. Capacity changes the boundary

Same code, different hidden-layer size. This is module 02 notebook 03's bias/variance
lesson, now in a neural network.
'''),
code('''
fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
for ax, n_hidden in zip(axes, [1, 16, 256]):
    p, _ = train(X_train, y_train, X_test, y_test,
                 epochs=1500, n_hidden=n_hidden, verbose=False)
    acc = accuracy(X_test, y_test, p)
    plot_boundary(p, X_test, y_test, ax, f"{n_hidden} hidden units - test acc {acc:.3f}")
plt.tight_layout()
plt.show()
'''),
md('''
One unit can only draw a straight line — underfitting. Sixteen units draw the curve.
256 units start carving out little islands around individual noisy points — overfitting,
exactly the degree-20 polynomial from module 02 notebook 03, wearing a different hat.

## 8. Check it against a library
'''),
code('''
from sklearn.neural_network import MLPClassifier

sk = MLPClassifier(hidden_layer_sizes=(16,), activation="relu",
                   max_iter=3000, random_state=42).fit(X_train, y_train)

print(f"our NumPy network : {accuracy(X_test, y_test, trained):.3f}")
print(f"sklearn MLP       : {sk.score(X_test, y_test):.3f}")
print("\\nSame algorithm. Ours is ~40 lines; theirs adds optimisers, batching and early stopping.")
'''),
md('''
---
## Exercises

1. Set the learning rate to `5.0`, then to `0.001`. Describe what each does to the loss
   curve, and name the failure mode in each case.
2. Replace ReLU with sigmoid in the hidden layer (you must also change the ReLU derivative
   in `backward` — it becomes `a * (1 - a)`). Does it still train? Is it slower?
3. Remove the activation entirely (`A1 = Z1`) and retrain. Plot the boundary. Explain the
   result in one sentence.
4. Add a **second** hidden layer. You will need to extend `init_params`, `forward` and
   `backward`. This is the exercise that proves you understood backprop.
5. **Stretch:** implement mini-batch gradient descent — shuffle the training data and update
   on batches of 32 instead of the full set each epoch. Compare convergence speed.

## Next

Notebook 02 rebuilds the perceptron in TensorFlow; notebook 06 does the same in PyTorch.
Both will feel small now — you already know what is underneath.
'''),
])
