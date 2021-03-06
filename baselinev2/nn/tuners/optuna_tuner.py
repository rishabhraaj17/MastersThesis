import torch
import torch.optim as optim
import torch.utils.data

import optuna
from torch.utils.data import random_split

from average_image.constants import SDDVideoClasses, SDDVideoDatasets
from baselinev2.config import ROOT_PATH
from baselinev2.constants import NetworkMode, SDDVideoClassAndNumbers
from baselinev2.nn.dataset import BaselineDataset, BaselineGeneratedDataset
from baselinev2.nn.models import BaselineRNNStacked, BaselineRNNStackedSimple
from baselinev2.nn.train import get_train_validation_dataset_for_class
from baselinev2.utils import social_lstm_parser
from baselinev2.nn.social_lstm.model import BaselineLSTM

DEVICE = torch.device("cuda")
DIR = f'{ROOT_PATH}Plots/Optuna'
EPOCHS = 20
LOG_INTERVAL = 10


# N_TRAIN_EXAMPLES = BATCHSIZE * 30
# N_VALID_EXAMPLES = BATCHSIZE * 10


def define_model(trial: optuna.Trial):
    use_batch_norm = trial.suggest_int("use_batch_norm", 0, 1)
    use_gru = trial.suggest_int("use_gru", 0, 1)
    rnn_layers = trial.suggest_int("rnn_layers", 1, 4)
    # return BaselineRNNStacked(use_batch_norm=bool(use_batch_norm), return_pred=True)
    # return BaselineLSTM(args=social_lstm_parser(pass_final_pos=False), generated_dataset=False,
    #                     use_batch_norm=use_batch_norm)
    return BaselineRNNStackedSimple(use_batch_norm=bool(use_batch_norm), use_gru=bool(use_gru), return_pred=True,
                                    learn_hidden_states=False, encoder_lstm_num_layers=rnn_layers,
                                    decoder_lstm_num_layers=rnn_layers, generated_dataset=True)


def get_loaders(trial):
    # dataset_train = BaselineDataset(SDDVideoClasses.LITTLE, 3, NetworkMode.TRAIN, meta_label=SDDVideoDatasets.LITTLE)
    # dataset_val = BaselineDataset(SDDVideoClasses.LITTLE, 3, NetworkMode.VALIDATION,
    #                               meta_label=SDDVideoDatasets.LITTLE)

    # dataset_train = BaselineGeneratedDataset(SDDVideoClasses.LITTLE, 3, NetworkMode.TRAIN,
    #                                          meta_label=SDDVideoDatasets.LITTLE)
    # dataset_val = BaselineGeneratedDataset(SDDVideoClasses.LITTLE, 3, NetworkMode.VALIDATION,
    #                                        meta_label=SDDVideoDatasets.LITTLE)

    dataset_train, dataset_val = get_train_validation_dataset_for_class(
        train_video_class=[SDDVideoClassAndNumbers.LITTLE, SDDVideoClassAndNumbers.DEATH_CIRCLE],
        train_meta_label=[SDDVideoDatasets.LITTLE, SDDVideoDatasets.DEATH_CIRCLE],
        val_video_class=[SDDVideoClassAndNumbers.LITTLE, SDDVideoClassAndNumbers.DEATH_CIRCLE,
                         SDDVideoClassAndNumbers.HYANG],
        val_meta_label=[SDDVideoDatasets.LITTLE, SDDVideoDatasets.DEATH_CIRCLE, SDDVideoDatasets.HYANG]
        , get_generated=True, videos_to_skip_for_train=[(), (), ()],
        videos_to_skip_for_val=[(), (), ()])

    # batch_size = trial.suggest_int("batch_size", 64, 1024, step=64)
    batch_size = 32
    # Load MNIST dataset.
    train_loader = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=True
    )
    valid_loader = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=True
    )

    return train_loader, valid_loader


def objective(trial: optuna.Trial):
    # Generate the model.
    model = define_model(trial).to(DEVICE)

    # Generate the optimizers.
    # optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "RMSprop"])
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam"])
    lr = trial.suggest_float("lr", 1e-5, 5e-5, log=True)
    optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr)

    # Get the MNIST dataset.
    train_loader, valid_loader = get_loaders(trial)

    all_epoch_loss = []
    # Training of the model.
    for epoch in range(EPOCHS):
        model.train()
        for batch_idx, data in enumerate(train_loader):
            # Limiting training data for faster epochs.
            # if batch_idx * BATCHSIZE >= N_TRAIN_EXAMPLES:
            #     break
            optimizer.zero_grad()

            data = [d.to(DEVICE) for d in data]

            loss, ade, fde, ratio, _ = model.one_step(data)  # model(data)
            loss.backward()
            optimizer.step()

        # Validation of the model.
        model.eval()
        correct = 0
        running_loss = []
        with torch.no_grad():
            for batch_idx, data in enumerate(valid_loader):
                # Limiting validation data.
                # if batch_idx * BATCHSIZE >= N_VALID_EXAMPLES:
                #     break
                data = [d.to(DEVICE) for d in data]
                loss, ade, fde, ratio, _ = model.one_step(data)  # model(data)
                # Get the index of the max log-probability.
                running_loss.append(loss.item())

        running_loss = torch.tensor(running_loss).mean()
        all_epoch_loss.append(running_loss)
        trial.report(running_loss, epoch)

        # Handle pruning based on the intermediate value.
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return torch.tensor(all_epoch_loss).mean()


if __name__ == "__main__":
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=200, timeout=1800, n_jobs=12, show_progress_bar=True)

    pruned_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    complete_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
