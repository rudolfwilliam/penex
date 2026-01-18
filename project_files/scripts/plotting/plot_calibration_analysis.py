import os

if 'ENTITY' not in os.environ:
    raise ValueError("ENTITY environment variable not set")

import pickle
import logging

import wandb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scienceplots


def plot_calibration_analysis(
                            plotting_dir,
                            wandb_project_name,
                            failed_states,
                            num_epochs,
                            y_range,
                            x_range,
                            y_ticks_args,
                            rhos,
                            name2rho,
                            gammas,
                            name2gamma,
                            epsilons,
                            name2epsilon,
                            alphas,
                            name2alpha,
                            file_name,
                            reload_data=False
                            ):

    file_path = os.path.join(plotting_dir, "tmp", "run_histories.pkl")

    api = wandb.Api()
    runs = api.runs(os.environ['ENTITY'] + "/" + wandb_project_name)

    if reload_data:
        # Pre-cache run histories
        run_histories = {}
        for run in runs:
            # Assume run.id is unique; adjust if needed
            run_histories[run.id] = pd.DataFrame(list(run.scan_history()))
        # save to disk
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'wb') as f:
            pickle.dump(run_histories, f)
        logging.info("Successfully loaded run histories from wandb and saved to disk.")
    else:
        # get from disk
        assert os.path.isfile(file_path), "No data has been saved. Please use --reload_data True."
        with open(file_path, 'rb') as f:
            run_histories = pickle.load(f)
        logging.info("Successfully loaded run histories from disk.")

    legend_entries = {}

    def plot_method(
            params, 
            name2param, 
            method_run_name, 
            num_dec, 
            label, 
            marker, 
            color_cycle_nr,
            yticks
            ):
        param_vals_ce = {"{:.{}f}".format(param, num_dec): [] for param in params}
        param_vals_ece = {"{:.{}f}".format(param, num_dec): [] for param in params}
        for run in runs:
            if run.state.lower() in failed_states:
                continue
            if run.name.startswith(method_run_name):
                # get name
                param_name = run.name.split("_")[-1]
                if param_name not in name2param.keys():
                    continue
                param_val = name2param[param_name]
                # get value
                try:
                    df = run_histories[run.id]
                    values_ce = np.where(df["train_loss_epoch"] == 'NaN', np.nan, df["train_loss_epoch"]).astype(np.float64)
                    values_ece = np.where(df["train_loss_epoch"] == 'NaN', np.nan, df["eval_ece"]).astype(np.float64)
                except KeyError:
                    values_ce = np.zeros(num_epochs)
                    values_ece = np.zeros(num_epochs)
                    logging.warning(f"KeyError: 'train_loss' or 'eval_ece' not found in run {run.name}. Skipping this run.")
                # Create filtered values (exclude NaN for plotting)
                filtered_values_ce = np.array([x for x in values_ce if not (isinstance(x, float) and np.isnan(x))])
                filtered_values_ece = np.array([x for x in values_ece if not (isinstance(x, float) and np.isnan(x))])
                if len(filtered_values_ce) < num_epochs:
                    continue
                # save final value
                param_vals_ce[param_val].append(filtered_values_ce[num_epochs-1])
                param_vals_ece[param_val].append(-filtered_values_ece[num_epochs-1])

        sorted_items_ce = sorted((float(k), v) for k, v in param_vals_ce.items())
        sorted_items_ece = sorted((float(k), v) for k, v in param_vals_ece.items())
        x_vals, y_vals_ce = zip(*sorted_items_ce)
        _, y_vals_ece = zip(*sorted_items_ece)
        y_mean_ce = np.array([np.array(x).mean() for x in y_vals_ce])
        # redefine x_vals in terms of difference in training loss to CE training
        x_vals = (y_mean_ce - y_mean_ce[0])
        y_mean_ece = np.array([np.array(x).mean() for x in y_vals_ece])
        y_std_ece = np.array([np.array(x).std() for x in y_vals_ece])
        # only show x values that are in the range
        max_idx = np.where(x_vals <= x_range[1])[0][-1]
        x_vals = x_vals[:(max_idx + 1)]
        y_mean_ece = y_mean_ece[:(max_idx + 1)]
        y_std_ece = y_std_ece[:(max_idx + 1)]
        # plot
        plt.ylim(y_range[0], y_range[1])
        plt.yticks(np.arange(y_ticks_args[0], y_ticks_args[1], y_ticks_args[2]), fontsize=14)
        plt.xlim(x_range[0], x_range[1])
        plt.xlabel("regularization strength")
        plt.ylabel("-ECE")
        if yticks:
            plt.tick_params(axis='y', labelrotation=90, labelsize=14)
        else:
            plt.tick_params(axis='y', labelleft=False, labelsize=14)
        plt.tick_params(axis='x', labelsize=14)
        plt.grid(True)
        plt.plot(x_vals, y_mean_ece, marker=marker, lw=3, color=color_cycle[color_cycle_nr], label=label)
        plt.fill_between(x_vals, y_mean_ece - y_std_ece, y_mean_ece + y_std_ece, color=color_cycle[color_cycle_nr], alpha=0.3)

        # stuff to plot legend
        if label not in legend_entries:
            handles, _ = plt.gca().get_legend_handles_labels()
            if handles:
                legend_entries[label] = handles[-1]

    plt.style.use(["science", "vibrant"])
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Computer Modern Roman']
    plt.rcParams['font.size'] = 24
    # retrieve the color cycle from the style
    color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
    fig = plt.figure(figsize=(5, 5))
    plot_method(
        params=rhos,
        name2param=name2rho,
        method_run_name="train_ce_entropy-penalty",
        num_dec=1,
        label="confidence penalty",
        marker="s",
        color_cycle_nr=1,
        yticks=True
        )
    plot_method(
        params=gammas,
        name2param=name2gamma,
        method_run_name="train_focal-loss",
        num_dec=1,
        label="focal loss",
        marker="d",
        color_cycle_nr=3,
        yticks=True
        )
    plot_method(
        params=epsilons,
        name2param=name2epsilon,
        method_run_name="train_ce_dummy_adam_smoothing",
        num_dec=2,
        label="label smoothing",
        marker="^",
        color_cycle_nr=2,
        yticks=True
        )
    plot_method(
        params=alphas,
        name2param=name2alpha,
        method_run_name="train_exp-loss",
        num_dec=3,
        label="PENEX",
        marker="o",
        color_cycle_nr=4,
        yticks=True
    )

    # Create unified fancy legend
    fig.legend(
        legend_entries.values(),
        legend_entries.keys(),
        loc='upper center',
        bbox_to_anchor=(0.515, 1.2),
        ncol=4,
        frameon=True,
        fancybox=True,
        fontsize=24
    )

    plt.savefig(os.path.join("plots", file_name + ".pdf"))   
