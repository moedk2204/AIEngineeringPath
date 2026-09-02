# Module 04 — Deep Learning

**Goal:** understand what a neural network actually does, then use TensorFlow and PyTorch
without either feeling like magic.

## Before you start

Finish module 02. Notebook 03 there (overfitting, learning curves) is the direct ancestor
of notebook 08 here, and notebook 01 here assumes you are comfortable with NumPy.

## Order

| # | Notebook | Focus | Needs |
|---|---|---|---|
| 00 | `00-when-to-use-deep-learning.ipynb` | **Should you use a neural net at all?** | sklearn |
| 01 | `01-neural-net-from-scratch-numpy.ipynb` | **Forward pass, backprop, gradient descent — by hand** | numpy |
| 02 | `02-perceptron-from-scratch.ipynb` | a single neuron, in TensorFlow | tensorflow |
| 03 | `03-intro-to-tensorflow.ipynb` | tensors, Keras layers, `model.fit` | tensorflow |
| 04 | `04-fashion-mnist-classification.ipynb` | first real classifier | tensorflow |
| 05 | `05-cnn-tensorflow.ipynb` | convolutions, pooling | tensorflow |
| 06 | `06-pytorch-fundamentals.ipynb` | **tensors, autograd, `nn.Module`, the training loop** | torch |
| 07 | `07-cnn-pytorch.ipynb` | the same CNN, in PyTorch | torch |
| 08 | `08-training-diagnostics-and-regularization.ipynb` | **dropout, weight decay, early stopping, LR** | torch |
| 09 | `09-transfer-learning.ipynb` | start from a pretrained model — what you will actually do | tensorflow |

Worked solutions for notebooks 02–05 and 07 are in `solutions/`.

## The four ideas this module exists to teach

1. **Most problems do not need deep learning.** Notebook 00 makes you prove it.
2. **Backprop is the chain rule, nothing more.** Notebook 01 makes you write it.
3. **Architecture encodes assumptions.** A CNN wins on images because it assumes
   translation invariance — not because it has more layers.
4. **Training diagnostics beat architecture tinkering.** Notebook 08: fix the learning
   rate and stop at the right epoch before you try anything clever.

## Setup

Notebooks 02–05 and 09 need TensorFlow; notebooks 06 and 08 need PyTorch. Notebooks 00 and
01 need neither.

```bash
pip install tensorflow torch torchvision
```

Both are large. If you only want the PyTorch track, install `torch` and do
00 → 01 → 06 → 07 → 08.

**No GPU?** Everything here runs on CPU. Notebooks 04, 05 and 09 will be slow but finish.
For anything bigger, use Google Colab (free GPU) — upload the notebook and the data folder.

## Self-check

- [ ] Given a dataset, argue for or against a neural network in three sentences.
- [ ] Write the forward pass of a two-layer network from memory.
- [ ] Explain why `optimizer.zero_grad()` exists and what breaks without it.
- [ ] Look at a loss curve and say whether the learning rate is too high, too low or fine.
- [ ] Explain why a CNN handles a shifted image and a fully-connected network does not.
- [ ] Name the order to try fixes when a model overfits.
