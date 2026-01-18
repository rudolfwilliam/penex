# Penalized Exponential Loss (PENEX)

Do you still train your neural networks using cross-entropy loss? How about you give [PENEX](https://arxiv.org/abs/2510.02107) a try, a new loss that is inspired by the infamous [AdaBoost algorithm](https://link.springer.com/chapter/10.1007/3-540-59119-2_166):

$$
\mathcal{L}_{\mathrm{\scriptscriptstyle PENEX}}(f; \alpha) =
\hat{\mathbb{E}} \Bigg\[
    \exp \Bigl( -\alpha f^{(y)}(\mathbf{x}) \Bigr) 
    + \rho(\alpha) \sum_{j=1}^K \exp \Bigl( f^{(j)}(\mathbf{x}) \Bigr)
\Bigg\].
$$

You are not convinced that you need PENEX? Maybe these CIFAR-100 results will change your mind:

<p align="center">
<img src="/assets/long_metric_curves_cifar100.svg" width="500">
</p>

Our implementation is based on [PyTorch](https://pytorch.org/).

## Setup :computer:

First, download the repository:
```bash
git clone https://github.com/rudolfwilliam/penex
```

Then, install the package via
```bash
pip install .
```
Now, let us have some fun! :rocket:

## Minimal Training and Inference Examples

The best part is that integrating PENEX into your training loop is almost no effort.

### Training

During training, just replace `nn.CrossEntropyLoss` by `PENEX`:

```python

import torch
import torch.nn as nn
import torch.optim as optim

from penex.losses import PENEX

criterion = PENEX() # PENEX instead of nn.CrossEntropyLoss

# Dummy dataset
X = torch.randn(100, 10)
y = torch.randint(0, 2, (100,))

# Simple model
model = nn.Sequential(
    nn.Linear(10, 50),
    nn.ReLU(),
    nn.Linear(50, 2)
)

optimizer = optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(10):
    optimizer.zero_grad()
    logits = model(X)
    loss = criterion(logits, y)    # Compute loss
    loss.backward()
    optimizer.step()
```

### Inference

Inference (that is, when you make predictions) is almost as simple as training.

>[!IMPORTANT]
> There is only one difference between cross-entropy and PENEX during inference: For PENEX, you need to perform inference at temperature
> $(1 + \alpha)^{-1}$, where $\alpha$ is the sensitivity parameter.

In code, this is just one extra line

```python
model.eval()

# New input sample
x_new = torch.randn(1, 10)

# Disable gradient calculation
with torch.no_grad():
    logits = model(x_new)
    logits_rescaled = logits * (1 + criterion.sensitivity) # IMPORTANT LINE
    probs = torch.softmax(logits_rescaled, dim=-1)
    pred_class = torch.argmax(probs, dim=-1)

print("Probabilities:", probs)
print("Predicted class:", pred_class.item())
```

That's it! Please check out `project_files/scripts/plotting/simple_2D_example.ipynb` for a slightly more extensive demo.

## Practical Advice

PENEX can be more sensitive than cross-entropy and may require a smaller learning rate. If training becomes unstable after replacing cross-entropy with PENEX, adjusting the learning rate should be your first step.

## Citation

If you find PENEX useful, we would be happy if you could leave our repository a star :star: and cite our pre-print :page_facing_up:. The bibtex entry is

```bibtex
@article{kladny2025penex,
  title={{PENEX: AdaBoost-Inspired Neural Network Regularization}},
  author={Kladny, Klaus-Rudolf and Sch{\"o}lkopf, Bernhard and Muehlebach, Michael},
  journal={arXiv preprint arXiv:2510.02107},
  year={2025}
}
```

## Reproducing Paper Results

If you would like to reproduce the experiments from [our paper](https://arxiv.org/abs/2510.02107), please take a look at our `EXPERIMENTS.md`.

## Problems, Questions or Feedback?

Please create an issue or inform me via e-mail: *kkladny [at] tuebingen [dot] mpg [dot] de*

<div align="center">

> "Mathematical theory is not critical to the development of machine learning. But scientific inquiry is."
> 
> — Leo Breiman
> 
</div>
