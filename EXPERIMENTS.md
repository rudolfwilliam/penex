## Reproducing Paper Results

### Code Organization

```
.
├── penex/                               # core functionality
│     ├── losses.py                      # implementation of losses
│     └── constraint_handlers.py         # various functions that keep logits from diverging (mostly interesting for ablation experiments)
│
├── project_files/
│         ├── vision/                    # vision experiments
│         │      ├── cifar10/
│         │      │      ├── configs/     # training parameters for each experiment
│         │      │      ├── models/      # experiment-specific model architectures
│         │      │      └── scripts/     # scripts for plotting, tuning and training models
│         │      └── ...
│         ├── nlp/                       # language experiment (BBC News)
│         │    └── ...
│         └── utils.py                   # various functions that are needed here and there
└── ...
```

### Setup

To reproduce the results from our paper, first create a conda environment from the `environment.yaml` file (assuming you name it `penex`)
```bash
conda env create -n penex -f environment.yaml
```
and activate it
```bash
conda activate penex
```

### Vision Experiments

### 1. Hyperparameter Tuning (CIFAR-10)
Run tuning for each loss variant. Optimal `eval_loss` values are printed to stdout.

| Loss Variant        | Command                                                                                      |
|---------------------|----------------------------------------------------------------------------------------------|
| cross-entropy  | `python -m project_files.vision.cifar10.scripts.tune_hparams --loss_func ce --constraint_handler dummy` |
| label smoothing     | `... --loss_func ce --constraint_handler dummy --cfg_extension smoothing`                                |
| confidence penalty        | `... --loss_func ce --constraint_handler entropy-penalty`                                               |
| focal loss          | `... --loss_func focal-loss --constraint_handler dummy`                                                 |
| **PENEX**           | `... --loss_func exp-loss --constraint_handler sumexp-penalty`                                       |

### 2. Training (CIFAR-10)
Use tuned parameters in `configs/*.json` (already filled).

| Loss Variant   | Command                                                                                  |
|----------------|------------------------------------------------------------------------------------------|
| cross-entropy             | `python -m project_files.vision.cifar10.scripts.train --loss_func ce --constraint_handler dummy --ckpt_name entropy --log_name entropy`    |
| label smoothing  | `... --loss_func ce --constraint_handler dummy --cfg_extension smoothing --ckpt_name smoothing --log_name smoothing`                            |
| confidence penalty   | `... --loss_func ce --constraint_handler entropy-penalty --ckpt_name entropy --log_name entropy`                                           |
| focal loss     | `... --loss_func focal-loss --constraint_handler dummy --ckpt_name focal --log_name focal`                                             |
| **PENEX**      | `... --loss_func exp-loss --constraint_handler sumexp-penalty --ckpt_name penex --log_name penex`                                   |

### 3. Calibration & Ablation Studies
Change `--constraint_handler` flag for each ablation:
- `augmented-lagrangian`
- `squared-penalty`
- `hard`
- `dummy`

Run with appropriate `--cfg_extension` and `--seed` flags as needed.

---

### Natural Language Processing Experiments (BBC News)

#### Hyperparameter Tuning (NLP)
Run tuning for each NLP loss variant. Optimal `eval_loss` values are printed to stdout.

| Loss Variant       | Command                                                                                                      |
|--------------------|--------------------------------------------------------------------------------------------------------------|
| confidence penalty       | `python -m project_files.nlp.bbc_news.scripts.tune_hparams --loss_func ce_entropy`                            |
| label smoothing     | `... --loss_func ce_smoothing`                                                                                |
| focal loss         | `... --loss_func focal-loss`                                                                                  |
| **PENEX**          | `... --loss_func penex`                                                                                      |

#### Training (NLP)
Use tuned parameters in `configs/*.json` (already filled).

| Loss Variant       | Command                                                                                                      |
|--------------------|--------------------------------------------------------------------------------------------------------------|
| cross-entropy       | `python -m project_files.nlp.bbc_news.scripts.train --loss_func ce`   
| confidence penalty       | `... --loss_func ce_entropy`                                  |
| label smoothing     | `... --loss_func ce_smoothing`                                                                                |
| focal loss         | `... --loss_func focal-loss`                                                                                  |
| **PENEX**          | `... --loss_func penex`                                                                                      |

---

### Plotting & Tables

```bash
# Validation curves (Fig. 2a, Appx. E.1)
python -m project_files.vision.cifar10.scripts.plotting.plot_metric_curves

# Ablation baselines (Fig. 6)
python -m project_files.vision.cifar10.scripts.plotting.plot_metric_curves --baselines ablations_poor
python -m project_files.vision.cifar10.scripts.plotting.plot_metric_curves --baselines ablations_good

# Calibration analysis (Fig. 2b)
python -m project_files.vision.cifar10.scripts.plotting.plot_calibration_analysis --reload_data

# Summary table (Tab. 1)
python -m project_files.scripts.print_table
```