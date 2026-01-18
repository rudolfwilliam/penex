import os
import json
import hashlib
import sys
from datetime import datetime

from torch.utils.data import DataLoader, random_split
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import wandb
import torch

from project_files.utils import set_seed, load_json, get_num_devices
from project_files.vision.scripts.utils import bootstrap_evaluate
from project_files.vision.base import parse_args
from project_files.vision.base import update_cfg_from_args, setup_data


def is_main_process():
    """Check if this is the main process (rank 0) in distributed training."""
    # Check various environment variables that indicate distributed training
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    global_rank = int(os.environ.get('RANK', 0))
    
    # For PyTorch Lightning, also check these
    node_rank = int(os.environ.get('NODE_RANK', 0))
    
    # Main process is when all ranks are 0
    return local_rank == 0 and global_rank == 0 and node_rank == 0


def create_config_hash(cfg, args):
    """Create a unique hash from the configuration and arguments to identify the run."""
    # Combine all relevant parameters into a single string
    config_dict = {
        'cfg': cfg,
        'loss_func': args.loss_func,
        'constraint_handler': args.constraint_handler,
        'optimizer': args.optimizer,
        'label_noise': args.label_noise,
        'num_classes': args.num_classes,
        'train_set_size': args.train_set_size,
        'seed': args.seed,
        'cfg_extension': args.cfg_extension,
        'name_extension': args.name_extension
    }
    
    # Convert to JSON string for consistent hashing
    config_str = json.dumps(config_dict, sort_keys=True)
    
    # Create SHA256 hash (first 16 chars for readability)
    config_hash = hashlib.sha256(config_str.encode()).hexdigest()[:16]
    
    return config_hash


def create_sentry_file(log_dir, wandb_project_name, cfg, args):
    """Create a sentry file that prevents duplicate runs."""
    if not is_main_process():
        print(f"Non-main process (rank {os.environ.get('LOCAL_RANK', 'unknown')}) - skipping sentry file creation")
        return None
    
    config_hash = create_config_hash(cfg, args)
    
    # Create filename with project name and config hash
    sentry_filename = f"RUNNING_{wandb_project_name}_{config_hash}.sentry"
    sentry_path = os.path.join(log_dir, sentry_filename)
    
    # Check if sentry file already exists
    if os.path.exists(sentry_path):
        print(f"ERROR: Sentry file already exists: {sentry_path}")
        print("This indicates a previous run with identical configuration is still running or crashed.")
        print("If you're sure no other job is running with this config, delete the sentry file and restart.")
        sys.exit(1)
    
    # Create sentry file with configuration info
    with open(sentry_path, "w") as f:
        f.write("=== JOB SENTRY FILE ===\n")
        f.write(f"Started at: {datetime.now()}\n")
        f.write(f"Config hash: {config_hash}\n")
        f.write(f"Project: {wandb_project_name}\n")
        f.write("\n=== CONFIGURATION ===\n")
        f.write(json.dumps(cfg, indent=2))
        f.write("\n\n=== ARGUMENTS ===\n")
        f.write(f"loss_func: {args.loss_func}\n")
        f.write(f"constraint_handler: {args.constraint_handler}\n")
        f.write(f"optimizer: {args.optimizer}\n")
        f.write(f"seed: {args.seed}\n")
        f.write(f"label_noise: {args.label_noise}\n")
        f.write(f"num_classes: {args.num_classes}\n")
        f.write(f"train_set_size: {args.train_set_size}\n")
    
    print(f"Created sentry file: {sentry_path}")
    return sentry_path


def remove_sentry_file(sentry_path):
    """Remove the sentry file after successful completion."""
    if not is_main_process() or sentry_path is None:
        return
    
    try:
        if os.path.exists(sentry_path):
            os.remove(sentry_path)
            print(f"Removed sentry file: {sentry_path}")
    except Exception as e:
        print(f"Warning: Could not remove sentry file {sentry_path}: {e}")


def train(
        *,
        dataset_train,
        dataset_test,
        model,
        wandb_project_name,
        cfg,
        vis_dir,
        log_dir,
        ckpt_dir,
        name,
        wandb_flag=True,
        test_flag=False
        ):
    set_seed(cfg["training_params"]["seed"])
    # Split into train and validation
    sr = cfg["training_params"]["split_ratio"]
    train_size = int(sr*len(dataset_train))
    val_size = len(dataset_train) - train_size
    train_dataset, val_dataset = random_split(
                                    dataset_train, 
                                    [train_size, val_size], 
                                    generator=torch.Generator().manual_seed(cfg["training_params"]["seed"])
                                    )
    if wandb_flag:
        # Initialize WandbLogger
        wandb_logger = WandbLogger(project=wandb_project_name, name=name)
        wandb_logger.log_hyperparams(cfg)
    callbacks = generate_callbacks(vis_dir, log_dir, ckpt_dir)
    # Create data loaders
    train_loader = DataLoader(
                        train_dataset, 
                        batch_size=cfg["training_params"]["batch_size"], 
                        shuffle=True, 
                        num_workers=cfg["training_params"]["num_workers"],
                        pin_memory=cfg["training_params"].get("pin_memory", False),
                        persistent_workers=cfg["training_params"].get("persistent_workers", False)
                        )
    val_loader = DataLoader(
                        val_dataset, 
                        batch_size=cfg["training_params"]["batch_size"], 
                        shuffle=False, 
                        num_workers=cfg["training_params"]["num_workers"],
                        pin_memory=cfg["training_params"].get("pin_memory", False),
                        persistent_workers=cfg["training_params"].get("persistent_workers", False)
                        )
    torch.set_float32_matmul_precision(cfg["training_params"]["precision"]) # medium or high precision
    trainer = Trainer(
        enable_checkpointing=True if ckpt_dir else False,
        max_epochs=cfg["training_params"]["max_epochs"], 
        val_check_interval=cfg["training_params"]["val_check_interval"],
        logger=wandb_logger if wandb_flag else True,
        callbacks=[callback for callback in callbacks if callback is not None],
        gradient_clip_val=cfg["training_params"]["gradient_clip_val"],
        gradient_clip_algorithm="norm",
        strategy="auto",
        accelerator="auto",
        devices="auto",
    )
    trainer.fit(model, train_loader, val_loader)

    if test_flag:
        data_loader = DataLoader(
                            dataset_test, 
                            batch_size=cfg["training_params"]["batch_size"]
                            )
        metrics = trainer.test(model=model, dataloaders=data_loader)[0]
        bootstrapped_metrics = bootstrap_evaluate(
                                    dataset=dataset_test, 
                                    trainer=trainer,
                                    model=model,
                                    batch_size=cfg["training_params"]["batch_size"]
                                    ) # evaluate with bootstrapping
        with open(log_dir + "/test_results.json", "w") as f:
            json.dump(metrics, f, indent=4)
        with open(log_dir + "/bootstrap_results.json", "w") as f:
            json.dump(bootstrapped_metrics, f, indent=4)

    if wandb_flag:
        wandb.finish()
    # relevant for hyper parameter tuning
    return trainer.callback_metrics["eval_loss"].item() 


def generate_callbacks(vis_dir, log_dir, ckpt_dir):
    callbacks = []
    # Define the ModelCheckpoint callback
    checkpoint_callback = ModelCheckpoint(
        dirpath=ckpt_dir,
        monitor="eval_loss",
        mode="min",
        save_top_k=1,                # Save only the best model
        save_last=True,               # Save the last checkpoint
        filename="best-checkpoint",
    ) if ckpt_dir else None
    callbacks.append(checkpoint_callback)

    return callbacks


def train_vision_model(
        config_dir,
        log_dir,
        vis_dir,
        ckpt_dir,
        wandb_project_name,
        overridable_params,
        load_data_func,
        create_model_func,
        test_flag=False
    ):

    args = parse_args()
    cfg = load_json(os.path.join(config_dir, args.cfg_name))

    # overwrite seed if supplied
    if args.seed is not None:
        cfg["training_params"]["seed"] = args.seed
    
    # Create sentry file to prevent duplicate runs
    sentry_path = create_sentry_file(log_dir, wandb_project_name, cfg, args)
    
    set_seed(cfg["training_params"]["seed"])

    # Update cfg with command-line arguments if provided
    update_cfg_from_args(cfg, args, overridable_params)

    train_dataset, test_dataset = load_data_func(
                                    label_noise=args.label_noise, 
                                    seed=cfg["training_params"]["seed"],
                                    num_classes=args.num_classes,
                                    size=args.train_set_size
                                    )

    # load model
    model = create_model_func(
                constraint_handler=args.constraint_handler, 
                loss_func=args.loss_func, 
                model_params=cfg["model_params"], 
                constraint_params=cfg["constraint_params"],
                max_epochs=cfg["training_params"]["max_epochs"],
                batch_size=cfg["training_params"]["batch_size"],
                warmup_epochs=cfg["training_params"].get("warmup_epochs", None),
                architecture=cfg.get("architecture", "convnet"),
                deep_supervision=cfg.get("deep_supervision", False),
                num_devices=get_num_devices(),
                dataset_size=train_dataset.__len__(),
                num_classes=args.num_classes
            )
    
    log_dir, vis_dir, ckpt_dir = setup_data(
                                        args=args, 
                                        model=model, 
                                        log_dir=log_dir,
                                        ckpt_dir=ckpt_dir, 
                                        vis_dir=vis_dir
                                        )

    train(
        model = model,
        dataset_train=train_dataset,
        dataset_test=test_dataset,
        wandb_project_name=wandb_project_name,
        vis_dir=vis_dir,
        log_dir=log_dir,
        ckpt_dir=ckpt_dir,
        name=args.name,
        cfg=cfg,
        test_flag=test_flag
        )
    
    # Training completed successfully - remove sentry file
    remove_sentry_file(sentry_path)
