import os

if 'ENTITY' not in os.environ:
    raise ValueError("ENTITY environment variable not set")

import wandb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scienceplots
from matplotlib.ticker import MaxNLocator, FuncFormatter


def detect_crash_epoch(df, metrics):
    """
    Detect the epoch where a run permanently crashes (any metric becomes NaN forever).
    
    Args:
        df: DataFrame with run history
        metrics: Dictionary of metrics to check
        
    Returns:
        int or None: Epoch where permanent crash occurs, or None if no permanent crash detected
    """
    # Convert all metric columns to numeric, replacing 'NaN' strings with np.nan
    metric_data = {}
    for metric in metrics.keys():
        if metric in df.columns:
            values = np.where(df[metric] == 'NaN', np.nan, df[metric]).astype(np.float64)
            metric_data[metric] = values
        else:
            return None  # Missing metric, can't determine crash
    
    # Find the minimum length across all metrics
    min_length = min(len(values) for values in metric_data.values())
    if min_length <= 1:
        return None
    
    # Check each metric individually for permanent NaN (NaN forever from some point)
    earliest_permanent_crash = None
    
    for metric, values in metric_data.items():
        # Find the first epoch where this metric becomes NaN and stays NaN forever
        for epoch in range(1, len(values)):
            if np.isnan(values[epoch]):
                # Check if ALL remaining epochs are also NaN
                remaining_values = values[epoch:]
                if np.all(np.isnan(remaining_values)):
                    print(f"    Metric {metric} permanently failed at epoch {epoch}")
                    if earliest_permanent_crash is None or epoch < earliest_permanent_crash:
                        earliest_permanent_crash = epoch
                    break  # Found permanent failure for this metric
    
    if earliest_permanent_crash is not None:
        print(f"    Permanent crash detected at epoch {earliest_permanent_crash}")
    
    return earliest_permanent_crash

def clean_crashed_run_data(df, metrics, crash_epoch):
    """
    Set ALL metric values to NaN from crash_epoch onwards (once any metric permanently fails).
    
    Args:
        df: DataFrame with run history
        metrics: Dictionary of metrics to check
        crash_epoch: Epoch from which to set ALL values to NaN
        
    Returns:
        dict: Dictionary of cleaned metric arrays
    """
    cleaned_data = {}
    
    for metric in metrics.keys():
        if metric in df.columns:
            values = np.where(df[metric] == 'NaN', np.nan, df[metric]).astype(np.float64)
            
            # Set ALL values to NaN from crash_epoch onwards, regardless of which metric originally failed
            if crash_epoch < len(values):
                values[crash_epoch:] = np.nan
            
            cleaned_data[metric] = values
        else:
            cleaned_data[metric] = np.array([])
    
    return cleaned_data

def plot_metric_curves(
        wandb_project_name,
        baselines,
        num_epochs,
        baselines_dict,
        project_name,
        metrics,
        invert,
        alphas,
        zorders,
        line_widths,
        fontsize=24
        ):
    api = wandb.Api()

    runs = api.runs(os.path.join(os.environ['ENTITY'], wandb_project_name))
    baselines_dict = baselines_dict[baselines]

    plt.style.use(["science", "vibrant"])
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Computer Modern Roman']
    plt.rcParams['font.size'] = fontsize

    # create plot with one subplot per metric
    _, axs = plt.subplots(1, len(metrics), figsize=(16, 4))
    for i, metric in enumerate(metrics.keys()):
        # create subplot
        ax = axs[i]
        ax.set_xlabel("epoch")
        ax.set_ylabel(metrics[metric])
        seen = []
        for method in baselines_dict.keys():
            for run in runs:
                if run.state == 'failed':
                    continue
                if run.name == method and run.name not in seen:
                    seen.append(run.name)
                    run_id = run.id
                    # Retrieve the specific run by its path
                    run_path = wandb_project_name + f"/{run_id}"
                    run = api.run(run_path)
                    # Access the run's history (metrics logged over time)
                    history_list = list(run.scan_history())
                    df = pd.DataFrame(history_list)
                    
                    # Detect if the run crashed and clean the data
                    crash_epoch = detect_crash_epoch(df, metrics)
                    if crash_epoch is not None:
                        print(f"  Run {run.name} crashed at epoch {crash_epoch}, cleaning data...")
                        cleaned_data = clean_crashed_run_data(df, metrics, crash_epoch)
                        values = cleaned_data[metric]
                    else:
                        values = np.where(df[metric] == 'NaN', np.nan, df[metric]).astype(np.float64)
                    
                    # Create filtered values (exclude NaN for plotting)
                    try:
                        filtered_values = np.array([x for x in values if not (isinstance(x, float) and np.isnan(x))])[:num_epochs]
                    except:
                        continue
                    if invert[metric]:
                        filtered_values = -filtered_values
                    ax.tick_params(axis='y', labelrotation=90, labelsize=14)
                    ax.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=4))
                    ax.tick_params(axis='x', labelsize=14)
                    ax.grid(True)
                    if baselines == "standard" or baselines == "standard_long":
                        ax.plot(
                            filtered_values, 
                            label = baselines_dict[run.name], 
                            alpha = alphas[run.name],
                            zorder = zorders[run.name],
                            linewidth = line_widths[run.name]
                        )
                    else: # plot normally
                        ax.plot(
                            filtered_values, 
                            label = baselines_dict[run.name],
                            linewidth = 2.0
                        )
    plt.tight_layout()
    plt.legend(loc='lower center', bbox_to_anchor=(-1.5, 1.0), frameon=True, columnspacing=0.5, fancybox=True, ncol=len(baselines_dict))
    plt.savefig(os.path.join("plots", baselines + "_metric_curves_" + project_name + ".pdf"))
