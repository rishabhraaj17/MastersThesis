import os
import shutil
from pathlib import Path

import torch
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities.cloud_io import load as pl_load
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler, PopulationBasedTraining
from ray.tune.integration.pytorch_lightning import TuneReportCallback, \
    TuneReportCheckpointCallback

from average_image.constants import SDDVideoClasses, SDDVideoDatasets
from baselinev2.config import ROOT_PATH, SAVE_BASE_PATH
from baselinev2.constants import NetworkMode
from baselinev2.nn.dataset import BaselineDataset
from baselinev2.nn.models import BaselineRNNStacked


def get_dataset(video_class: SDDVideoClasses, video_number: int, mode: NetworkMode, meta_label: SDDVideoDatasets):
    return BaselineDataset(video_class, video_number, mode, meta_label=meta_label,
                           root='/home/rishabh/Thesis/TrajectoryPredictionMastersThesis/Datasets/SDD_Features/')


def get_datasets():
    train_dataset = get_dataset(video_class=SDDVideoClasses.LITTLE, video_number=3, mode=NetworkMode.TRAIN,
                                meta_label=SDDVideoDatasets.LITTLE)
    val_dataset = get_dataset(video_class=SDDVideoClasses.LITTLE, video_number=3, mode=NetworkMode.VALIDATION,
                              meta_label=SDDVideoDatasets.LITTLE)
    return train_dataset, val_dataset


class TuneBaselineRNNStacked(BaselineRNNStacked):

    def __init__(self, config, data_dir=f'{ROOT_PATH}Plots/Ray/', use_batch_norm=False, train_dataset=None,
                 val_dataset=None, num_workers=0, shuffle=True):
        super(TuneBaselineRNNStacked, self).__init__(use_batch_norm=use_batch_norm, num_workers=num_workers,
                                                     shuffle=shuffle, train_dataset=train_dataset,
                                                     val_dataset=val_dataset)

        self.data_dir = data_dir or os.getcwd()
        Path(data_dir).mkdir(parents=True, exist_ok=True)

        self.lr = config['lr']
        self.batch_size = config['batch_size']

    def validation_step(self, batch, batch_idx):
        loss, ade, fde, ratio = self.one_step(batch)
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('val/ade', ade * ratio, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('val/fde', fde * ratio, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return {'val_loss': loss, 'val_ade': ade * ratio, 'val_fde': fde * ratio}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        ade_list = [x["val_ade"] for x in outputs]
        avg_ade = sum(ade_list) / len(ade_list)
        fde_list = [x["val_fde"] for x in outputs]
        avg_fde = sum(fde_list) / len(fde_list)
        self.log("ptl/val_loss", avg_loss)
        self.log("ptl/val_ade", torch.tensor(avg_ade))
        self.log("ptl/val_fde", torch.tensor(avg_ade))


def train(config):
    train_dataset, val_dataset = get_datasets()
    model = TuneBaselineRNNStacked(config=config, train_dataset=train_dataset, val_dataset=val_dataset, num_workers=0,
                                   use_batch_norm=False, shuffle=True)
    trainer = Trainer(max_epochs=10, progress_bar_refresh_rate=0)

    trainer.fit(model)


def train_tune(config, data_dir=f'{ROOT_PATH}Plots/Ray/', num_epochs=10, num_gpus=0):
    train_dataset, val_dataset = get_datasets()
    model = TuneBaselineRNNStacked(config=config, train_dataset=train_dataset, val_dataset=val_dataset, num_workers=12,
                                   use_batch_norm=False, shuffle=True, data_dir=data_dir)
    trainer = Trainer(
        max_epochs=num_epochs,
        gpus=num_gpus,
        logger=TensorBoardLogger(
            save_dir=tune.get_trial_dir(), name="", version="."),
        progress_bar_refresh_rate=0,
        callbacks=[
            TuneReportCallback(
                {
                    "loss": "ptl/val_loss",
                    "ade": "ptl/val_ade"
                },
                on="validation_end")
        ])
    trainer.fit(model)


def tune_asha(num_samples=10, num_epochs=10, gpus_per_trial=0, cpu_per_trail=1):
    data_dir = os.path.join(f'{ROOT_PATH}Plots/Ray/', "mnist_data_")
    Path(data_dir).mkdir(parents=True, exist_ok=True)

    config = {
        "lr": tune.loguniform(1e-6, 1e-1),
        "batch_size": tune.choice([128, 256, 512, 1024, 2048]),
    }

    scheduler = ASHAScheduler(
        max_t=num_epochs,
        grace_period=1,
        reduction_factor=2)

    reporter = CLIReporter(
        parameter_columns=["lr", "batch_size"],
        metric_columns=["loss", "ade", "training_iteration"])

    analysis = tune.run(
        tune.with_parameters(
            train_tune,
            data_dir=data_dir,
            num_epochs=num_epochs,
            num_gpus=gpus_per_trial),
        resources_per_trial={
            "cpu": cpu_per_trail,
            "gpu": gpus_per_trial
        },
        metric="loss",
        mode="min",
        config=config,
        num_samples=num_samples,
        scheduler=scheduler,
        progress_reporter=reporter,
        name="tune_asha")

    print("Best hyperparameters found were: ", analysis.best_config)

    shutil.rmtree(data_dir)


def train_tune_checkpoint(config, checkpoint_dir=None, data_dir=None, num_epochs=10, num_gpus=0):
    train_dataset, val_dataset = get_datasets()
    trainer = Trainer(
        max_epochs=num_epochs,
        gpus=num_gpus,
        logger=TensorBoardLogger(
            save_dir=tune.get_trial_dir(), name="", version="."),
        progress_bar_refresh_rate=0,
        callbacks=[
            TuneReportCheckpointCallback(
                metrics={
                    "loss": "ptl/val_loss",
                    "ade": "ptl/val_ade"
                },
                filename="checkpoint",
                on="validation_end")
        ])
    if checkpoint_dir:
        # Currently, this leads to errors:
        # model = LightningMNISTClassifier.load_from_checkpoint(
        #     os.path.join(checkpoint, "checkpoint"))
        # Workaround:
        ckpt = pl_load(
            os.path.join(checkpoint_dir, "checkpoint"),
            map_location=lambda storage, loc: storage)
        model = TuneBaselineRNNStacked._load_model_state(
            ckpt, config=config, data_dir=data_dir)
        trainer.current_epoch = ckpt["epoch"]
    else:
        model = TuneBaselineRNNStacked(config=config, train_dataset=train_dataset, val_dataset=val_dataset,
                                       num_workers=12, use_batch_norm=False, shuffle=True, data_dir=data_dir)

    trainer.fit(model)


def tune_pbt(num_samples=10, num_epochs=10, gpus_per_trial=0, cpu_per_trail=1):
    data_dir = os.path.join(f'{ROOT_PATH}Plots/Ray/', "mnist_data_")
    Path(data_dir).mkdir(parents=True, exist_ok=True)

    config = {
        "lr": 1e-3,
        "batch_size": 64,
    }

    scheduler = PopulationBasedTraining(
        perturbation_interval=4,
        hyperparam_mutations={
            "lr": tune.loguniform(1e-6, 1e-1),
            "batch_size": [128, 256, 512, 1024, 2048]
        })

    reporter = CLIReporter(
        parameter_columns=["lr", "batch_size"],
        metric_columns=["loss", "ade", "training_iteration"])

    analysis = tune.run(
        tune.with_parameters(
            train_tune_checkpoint,
            data_dir=data_dir,
            num_epochs=num_epochs,
            num_gpus=gpus_per_trial),
        resources_per_trial={
            "cpu": cpu_per_trail,
            "gpu": gpus_per_trial
        },
        metric="loss",
        mode="min",
        config=config,
        num_samples=num_samples,
        scheduler=scheduler,
        progress_reporter=reporter,
        name="tune_pbt")

    print("Best hyperparameters found were: ", analysis.best_config)

    shutil.rmtree(data_dir)


if __name__ == '__main__':
    config = {
        "lr": 1e-3,
        "batch_size": 64
    }
    # train(config=config)
    tune_pbt(num_samples=20, num_epochs=10, gpus_per_trial=1, cpu_per_trail=12)
