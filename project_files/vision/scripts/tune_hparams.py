import os
from functools import partial

import optuna
from optuna.samplers import TPESampler
from optuna.storages import RDBStorage

from project_files.utils import set_seed, load_json
from project_files.vision.scripts.train import train
from project_files.vision.base import parse_args
from project_files.vision.base import SEARCH_SPACES
from project_files.vision.base import update_cfg_from_args

PARAM_LIST = ["model_params", "constraint_params"]


def objective(
              trial, 
              train_dataset,
              wandb_project_name,
              search_spaces,
              create_classifier_func,
              args,
              cfg
              ):
    
    for params in PARAM_LIST:
        for key, value in search_spaces[args.name][params].items():
            cfg[params][key] = suggest_param(trial, key, value[0], value[1])
    
    model = create_classifier_func(
                constraint_handler=args.constraint_handler,
                loss_func=args.loss_func,
                model_params=cfg["model_params"],
                architecture=cfg.get("architecture", "convnet"),
                deep_supervision=cfg.get("deep_supervision", False),
                constraint_params=cfg["constraint_params"]
            )

    try:
        criterion = train(
                        model=model,
                        dataset_train=train_dataset,
                        dataset_test=None, # not required
                        wandb_project_name=wandb_project_name,
                        vis_dir=None,
                        ckpt_dir=None,
                        log_dir=None,
                        cfg=cfg,
                        name="nr_" + str(trial.number) + "_" + \
                            "_".join([key + "_" + str(round(trial.params[key], ndigits=5)) for key in trial.params.keys()])
                        )
    except Exception as e:
        print(f"Trial failed with error: {e}")
        # Gracefully mark the trial as failed
        return float("inf")

    return criterion


def suggest_param(trial, key, value0, value1):
    if isinstance(value0, int) and isinstance(value1, int):
        return trial.suggest_int(key, value0, value1)
    elif isinstance(value0, float) and isinstance(value1, float):
        return trial.suggest_float(key, value0, value1)
    else:
        raise ValueError("Value types must be the same")


def tune_vision_model(
        config_dir,
        wandb_project_root,
        overridable_params,
        load_data_func,
        create_classifier_func,
        study_name,
        storage_url,
        n_trials
):
    args = parse_args()
    cfg = load_json(os.path.join(config_dir, args.cfg_name))

    set_seed(cfg["training_params"]["seed"])
    
    # Load cifar100 dataset
    train_dataset, _ = load_data_func(label_noise=args.label_noise)
    # Add the models parameters to the config
    cfg["model_params"]["optimizer"] = args.optimizer
    # Update cfg with command-line arguments if provided
    update_cfg_from_args(cfg, args, overridable_params)
    # Note: parameters that are not set in the search space dir 
    # will be set to the default values in the config file

    # create objective function
    cifar100_objective = partial(
                            objective,
                            train_dataset=train_dataset,
                            wandb_project_name=wandb_project_root + args.name,
                            search_spaces=SEARCH_SPACES,
                            create_classifier_func=create_classifier_func,
                            args=args,
                            cfg=cfg
                            )
    
    storage = RDBStorage(
        url=storage_url,
        heartbeat_interval=30,                    # more frequent pings
        grace_period=120,                         # mark stale RUNNING after 2 min
        engine_kwargs={
            "connect_args": {"timeout": 60},      # longer timeout for SQLite
            "pool_pre_ping": True,                # verify connections
            "pool_recycle": 3600                  # recycle connections hourly
        },
    )
    
    # Check if study exists first
    try:
        existing_studies = optuna.get_all_study_summaries(storage)
        study_exists = any(s.study_name == study_name for s in existing_studies)
    except:
        study_exists = False
    
    if study_exists:
        # Load existing study - don't specify sampler, let Optuna restore it
        print(f"Loading existing study: {study_name}")
        study = optuna.load_study(
            study_name=study_name,
            storage=storage
        )
    else:
        # Create new study with fresh sampler
        print(f"Creating new study: {study_name}")
        sampler = TPESampler(seed=cfg["training_params"]["seed"])
        study = optuna.create_study(
            direction="minimize", 
            sampler=sampler,
            study_name=study_name,
            storage=storage
        )
    
    print(f"Study loaded: {len(study.trials)} total trials")
    
    # Check for incomplete trials that should be retried
    incomplete_trials = [t for t in study.trials if t.state in [
        optuna.trial.TrialState.RUNNING,
        optuna.trial.TrialState.PRUNED,
        optuna.trial.TrialState.FAIL
    ]]
    
    if incomplete_trials:
        print(f"Found {len(incomplete_trials)} incomplete trials from previous crashes:")
        for trial in incomplete_trials:
            print(f"  Trial {trial.number}: {trial.params}")
        
        # Retry each incomplete trial with the same trial number and parameters
        for trial in incomplete_trials:
            print(f"Retrying trial {trial.number} with same parameters: {trial.params}")
            
            try:
                # Create a mock trial object that matches the original trial
                class MockTrial:
                    def __init__(self, original_trial):
                        self.number = original_trial.number
                        self.params = original_trial.params.copy()
                        self._trial_id = original_trial._trial_id
                        self.study = study
                        
                    def suggest_float(self, name, low, high, **kwargs):
                        return self.params[name]
                    
                    def suggest_int(self, name, low, high, **kwargs):
                        return self.params[name]
                    
                    def suggest_categorical(self, name, choices, **kwargs):
                        return self.params[name]
                    
                    def report(self, value, step=None):
                        pass  # Skip intermediate reporting for retries
                    
                    def should_prune(self):
                        return False  # Don't prune retries
                
                mock_trial = MockTrial(trial)
                
                # Run the objective with the original trial's parameters
                result = cifar100_objective(mock_trial)
                
                # Use study.tell() with the trial number to update it
                study.tell(trial.number, result, state=optuna.trial.TrialState.COMPLETE)
                
                print(f"  Successfully completed trial {trial.number}, result: {result}")
                
            except Exception as e:
                print(f"  Retry of trial {trial.number} failed: {e}")
                # Mark as failed
                study._storage.set_trial_state_values(trial._trial_id, optuna.trial.TrialState.FAIL)
        
        # Reload study to get updated trial states
        study = optuna.load_study(study_name=study_name, storage=storage)
    
    # Count completed trials (after retries)
    completed_trials = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    
    if completed_trials >= n_trials:
        print(f"Study already completed! {completed_trials}/{n_trials} trials finished")
        print(f"Best value: {study.best_value}")
        print(f"Best params: {study.best_params}")
    else:
        remaining_trials = n_trials - completed_trials
        print(f"Found {completed_trials}/{n_trials} completed trials")
        print(f"Running {remaining_trials} more trials to finish the study")
        
        # Continue with normal optimization
        study.optimize(cifar100_objective, n_trials=remaining_trials)
