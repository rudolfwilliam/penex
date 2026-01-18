import os
import argparse
import json
from functools import partial

from datasets import load_dataset, ClassLabel

from project_files.utils import load_json
from project_files.nlp.bbc_news.paths import CONFIG_DIR
from project_files.nlp.bbc_news.utils import preprocess
from project_files.nlp.bbc_news.meta_data import ID2LABEL


def get_args():
    parser = argparse.ArgumentParser(description="Run the evaluation module with given parameters") 
    parser.add_argument('--config', type=str, default=None, help='Path to configuration JSON file')
    #parser.add_argument('--ckpt_path', type=str, default=None, help='Optional checkpoint path to use for initialization')
    parser.add_argument('--loss_func', type=str, default="ce", help='name of the method', choices=["penex", "ce", "ce_smoothing", "ce_entropy", "focal-loss"])
    args = parser.parse_args()

    if args.config:
        try:
            with open(args.config, "r") as f:
                _ = json.load(f)
            config_path = args.config
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Error loading config file: {e}")
            exit(1)
    else:
        config_path = os.path.join(CONFIG_DIR, args.loss_func) + ".json"

    cfg = load_json(config_path)

    return args, cfg

def load_data(
        seed,
        tokenizer,
        num_training_examples
    ):
    dataset = load_dataset("SetFit/bbc-news")    
    # Apply tokenization to the dataset
    preprocess_ = partial(preprocess, tokenizer=tokenizer)
    encoded_dataset = dataset.map(preprocess_, batched=True)
    # Remove unnecessary columns
    encoded_dataset = encoded_dataset.remove_columns(["text", "label_text"])

    class_label = ClassLabel(names=list(ID2LABEL.values()))

    # Cast the 'label' column to ClassLabel. 
    casted_dataset = encoded_dataset.cast_column("label", class_label)
    split_dataset = casted_dataset["train"].train_test_split(
                                                            test_size=0.2,
                                                            seed=seed,
                                                            stratify_by_column="label"
                                                            )
    # Get train and validation splits
    train_dataset = split_dataset["train"].select(range(num_training_examples))
    val_dataset = split_dataset["test"] # validation set
    test_dataset = casted_dataset["test"]

    return train_dataset, val_dataset, test_dataset

def log_GPU_info():
    import torch
    # GPU Detection and Logging
    print("=" * 60)
    print("GPU DETECTION AND USAGE INFORMATION")
    print("=" * 60)
    
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        # Get CUDA version using different method
        import subprocess
        try:
            result = subprocess.run(['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader,nounits'], 
                                  capture_output=True, text=True)
            cuda_info = result.stdout.strip() if result.returncode == 0 else "nvidia-smi failed"
        except:
            cuda_info = "nvidia-smi not available"
        
        print(f"GPU driver info: {cuda_info}")
        print(f"Number of GPUs detected: {torch.cuda.device_count()}")
        print(f"Current GPU device: {torch.cuda.current_device()}")
        print(f"GPU device name: {torch.cuda.get_device_name()}")
        
        # Check GPU memory
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"  Memory allocated: {torch.cuda.memory_allocated(i) / 1e9:.2f} GB")
            print(f"  Memory cached: {torch.cuda.memory_reserved(i) / 1e9:.2f} GB")
            print(f"  Total memory: {torch.cuda.get_device_properties(i).total_memory / 1e9:.2f} GB")
    else:
        print("WARNING: CUDA is not available! Training will run on CPU.")
        print("This will be extremely slow for transformer models.")
    
    # Check environment variables that might affect GPU usage
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        print(f"CUDA_VISIBLE_DEVICES: {os.environ['CUDA_VISIBLE_DEVICES']}")
    else:
        print("CUDA_VISIBLE_DEVICES: Not set")
    
    print("=" * 60)
    print()


# search spaces for hyperparameters of different methods
SEARCH_SPACES = {
    "penex" : {
        "model_params": {
            "sensitivity" : [0.0, 1.0]
        },
        "hf_params":{}
    },
    "ce_entropy" : {
        "model_params": {
            "rho" : [0.0, 10.0]
        },
        "hf_params":{}
    },
    "ce_smoothing" : {
        "model_params": {},
        "hf_params": {
            "label_smoothing_factor" : [0.0, 1.0]
        }
    },
    "focal-loss" : {
        "model_params": {
            "gamma" : [0.0, 5.0]
        },
        "hf_params":{}
    }
}
