from typing import List, Tuple
import hydra
import pyrootutils
import lightning as L
from lightning import LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig
import numpy as np
import pandas as pd
import os
import torch

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src import utils
from src.interpretability.robustness_timeseries import Robustness
from src.interpretability.components.visualization import mean_vis, ridge_line_vis, skew_kurt_vis, frac_vis

log = utils.get_pylogger(__name__)

methods4eval = [
        "integrated_gradients",
        "deeplift",
        "deepliftshap",
        "gradshap",
        "kernelshap",
        "shapleyvalue",
    ]


@utils.task_wrapper
def evaluate_interpretability(cfg: DictConfig) -> Tuple[dict, dict]:
    """Evaluates interpretability methods for given checkpoint on a datamodule testset.
    """
    # set seed for random number generators in pytorch, numpy and python.random
    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)

    assert cfg.ckpt_path

    log.info(f"Instantiating datamodule <{cfg.datamodule._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.datamodule)

    datamodule.prepare_data()
    cfg.model.net.input_size = datamodule.input_size
    cfg.model.net.output_size = datamodule.num_classes

    datamodule.setup("test")
    test_loader = datamodule.test_dataloader()
    signal_list = []
    target_list = []

    for batch in test_loader:
            # the first key is the signal and the second is the target
            signal, target = batch
            signal_list.append(signal)
            target_list.append(target)
    np_signal = torch.cat(signal_list, dim=0).numpy()
    np_target = torch.cat(target_list, dim=0).numpy()
    # np_nounid = pd.read_csv(os.path.join(datamodule.data_dir, "targets.csv"))['noun_id'].tail(np_target.shape[0]).values
    np_nounid = np.array([f"sample_{i}" for i in range(np_target.shape[0])])

    # if save signal. target and nounid
    with open(os.path.join(cfg.paths.output_dir,"signal.npy"), 'wb') as f:
        np.save(f, np_signal)
    with open(os.path.join(cfg.paths.output_dir,"target.npy"), 'wb') as f:
        np.save(f, np_target)
    with open(os.path.join(cfg.paths.output_dir,"nounid.npy"), 'wb') as f:
        np.save(f, np_nounid)


    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model)

    log.info("Instantiating loggers...")
    logger: List[Logger] = utils.instantiate_loggers(cfg.get("logger"))

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer, logger=logger)

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "logger": logger,
        "trainer": trainer,
    }

    if logger:
        log.info("Logging hyperparameters!")
        utils.log_hyperparameters(object_dict)

    log.info("Starting testing!")
    trainer.test(model=model, datamodule=datamodule, ckpt_path=cfg.ckpt_path)

    metric_dict = trainer.callback_metrics

    # evaluate interpretability methods
    # if load signal, target and nounid from files
    np_signal = np.load(os.path.join(cfg.paths.output_dir,"signal.npy"))
    np_target = np.load(os.path.join(cfg.paths.output_dir,"target.npy"))
    np_nounid = np.load(os.path.join(cfg.paths.output_dir,"nounid.npy"), allow_pickle=True)

    corruption_percents = [0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95, 1.0]
    
    path_results = cfg.paths.output_dir

    robust = Robustness(model, np_signal, np_target, np_nounid, path_results)

    for method_name in methods4eval:
        robust.get_relevance(method_name)
    for method_name in methods4eval:
        _ = robust.get_corrupted_scores(corruption_percents, method_name)
        robust.summary_robust(method_name)
    robust.get_evaluation_metrics()
    
    # visualize results
    vis_path = os.path.join(path_results, "interpretability_evaluation")
    mean_vis(vis_path, methods4eval)
    ridge_line_vis(vis_path, methods4eval)
    skew_kurt_vis(vis_path, corruption_percents, methods4eval)
    frac_vis(vis_path, corruption_percents, methods4eval)

    return metric_dict, object_dict


@hydra.main(version_base="1.3", config_path="../configs", config_name="eval_interpret.yaml")
def main(cfg: DictConfig) -> None:
    # apply extra utilities
    # (e.g. ask for tags if none are provided in cfg, print cfg tree, etc.)
    utils.extras(cfg)

    evaluate_interpretability(cfg)


if __name__ == "__main__":
    main()
