import os
import argparse

import numpy as np

from project_files.utils import load_json


LOG_DIRS = { # standard evaluation metrics; no ROUGE, BLEU, etc.
    "CIFAR10" : "project_files/vision/cifar10/logs",
    "Noisy CIFAR10" : "project_files/vision/cifar10_noise_01/logs",
    "CIFAR100" : "project_files/vision/cifar100/logs",
    "PathMNIST" : "project_files/vision/pathMNIST/logs",
    "Imagenet" : "project_files/vision/imagenet/logs",
    "BBC News" : "project_files/nlp/bbc_news/logs"
}

METRIC_MAPPING = {
    "ACC" : "eval_accuracy",
    "-ECE" : "eval_ece",
    "-CE" : "eval_loss",
    "-BRIER" : "eval_brier_score"
}

INVERT = {
    "ACC" : False,
    "-ECE" : True,
    "-CE" : True,
    "-BRIER" : True
}

METHOD_MAPPING = {
    "ce" : "CE",
    "smoothing" : "\\makecell{label \\\\ smoothing}",
    "entropy" : "\\makecell{confidence \\\\ penalty}",
    "focal" : "\\makecell{focal \\\\ loss}",
    "penex" : "\\underline{PENEX}"
}

EXCLUDED_METHODS = []

def invert(val):
    return - val

def main(val_opt=False):
    aggr_results = {}
    for dataset_name, data_dir in LOG_DIRS.items():
        results = {}
        for dir in os.listdir(data_dir):
            if dir not in METHOD_MAPPING.keys():
                continue
            method_name = METHOD_MAPPING[dir]
            if method_name in EXCLUDED_METHODS:
                continue
            dir_path = os.path.join(data_dir, dir)
            appx = "_val_opt" if val_opt else ""
            means = load_json(dir_path + "/test_results" + appx + ".json")
            bootstrap_results = load_json(dir_path + "/bootstrap_results" + appx + ".json")
            results[method_name] = compute_mean_and_std(means, bootstrap_results)
        
        # Rearrange results to have method name before dataset name
        for method, metrics in results.items():
            if method not in aggr_results:
                aggr_results[method] = {}
            aggr_results[method][dataset_name] = metrics
    
    # Reorder aggr_results based on METHOD_MAPPING
    method_order = list(METHOD_MAPPING.values())  # ["CE", "\\makecell{label \\\\ smoothing}", ...]
    aggr_results_reordered = {method: aggr_results[method] for method in method_order if method in aggr_results}
    max_values = find_maximums(aggr_results_reordered)
    latex_table = generate_latex_table(aggr_results_reordered, max_values)
    print(latex_table)


def compute_mean_and_std(means, bootstrap_results):
    return {
        metric : f"$ {means[METRIC_MAPPING[metric]] if not INVERT[metric] else \
                      invert(means[METRIC_MAPPING[metric]]):.3f} \\pm {np.std(bootstrap_results[METRIC_MAPPING[metric]]):.3f}$" \
                      for metric in METRIC_MAPPING.keys()
    }


def find_maximums(aggr_results):
    max_values = {dataset: {metric: float('-inf') for metric in METRIC_MAPPING.keys()} for dataset in LOG_DIRS.keys()}
    for method_results in aggr_results.values():
        for dataset, dataset_results in method_results.items():
            for metric, value in dataset_results.items():
                if metric in max_values[dataset]:
                    if value == "$ - $":
                        continue
                    mean_value = float(value.split(' ')[1])
                    if mean_value > max_values[dataset][metric]:
                        max_values[dataset][metric] = mean_value
    return max_values


def generate_latex_table(aggr_results, min_values):
    header = (
        "\\begin{table}[tb]\n"
        "\\centering\n"
        "\\caption{\\textbf{Test Set Performance.} \\underline{Larger means better, best is bold}. Mean test result $\\pm 1$ standard deviation after $200$ training epochs. Standard deviation is obtained by running $100$ bootstrap evaluations~\\citep{tibshirani1993bootstrap}. Each method's hyperparameters are tuned on the validation set.}\n"
        "\\vspace{0.3em}\n"
        "\\label{tab:test_performance}\n"
        "\\resizebox{0.99 \\textwidth}{!}{\n"
        "\\begin{tabular}{ c || c || c | c | c | c | c | c }\n"
        "\\toprule\n"
        "\\textbf{Method} & \\textbf{Metric} & \\textbf{CIFAR-10} & \\textbf{Noisy CIFAR-10} & \\textbf{CIFAR-100} & \\textbf{PathMNIST} & \\textbf{ImageNet} & \\textbf{BBC News} \\\\\n"
        "\\midrule\n"
    )
    content = ""

    for method, datasets in aggr_results.items():
        content += f"\\multirow{{4}}{{*}}{{{method}}}\n"
        for _, metric in enumerate(METRIC_MAPPING.keys()):
            content += " & "
            content += f"{metric} "
            for dataset in LOG_DIRS.keys():
                result = datasets.get(dataset, {}).get(metric, '')
                if result != '':
                    if result == "$ - $":
                        content += "& $ - $ "
                        continue
                    mean_value = float(result.split(' ')[1])
                    if mean_value == min_values[dataset][metric]:
                        result = f"$ \\bf{{{float(result.split(' ')[1]):.3f}}} \\pm " + result.split(' ')[3]
                content += f"& {result} "
            content += "\\\\\n"
        content += "\\midrule\n" if method != list(aggr_results.keys())[-1] else "\\bottomrule\n"

    footer = (
        "\\end{tabular}\n"
        "}\n"
        "\\end{table}"
    )

    return header + content + footer


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run the evaluation module with given parameters")
    parser.add_argument('--val_opt', type=bool, default=False, help='Whether to load results at validation optimum.')
    args = parser.parse_args()
    main(args.val_opt)
