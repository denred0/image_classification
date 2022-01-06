import timm
import torch
import copy
import gc
import os
import numpy as np
import time
import torch.nn.functional as F
import matplotlib.pyplot as plt

from torch import nn, optim
from torch.cuda import amp
from torch.optim import lr_scheduler

from tqdm import tqdm
from sklearn.metrics import accuracy_score
from collections import defaultdict
from pathlib import Path

import config, dataset
from train_val_test_split import train_val_test_split

from my_utils import seed_everything, get_model


def fetch_scheduler(optimizer):
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.T_max[0], eta_min=config.min_lr[0])

    if config.scheduler == 'CosineAnnealingLR':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.T_max[0], eta_min=config.min_lr[0])
    elif config.scheduler == 'CosineAnnealingWarmRestarts':
        scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=config.T_0[0], eta_min=config.min_lr[0])
    elif config.scheduler == 'ExponentialLR':
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99, verbose=False)
    elif config.scheduler is None:
        return None

    return scheduler


def train_one_epoch(model, loader, loss_fn, optimizer, scheduler, device, epoch):
    model.train()

    dataset_size = 0
    running_loss = 0.0

    TARGETS = []
    PREDS = []

    epoch_loss = 0
    scaler = amp.GradScaler()

    bar = tqdm(enumerate(loader), total=len(loader))
    for batch_idx, (data, targets) in bar:
        data = data.to(device)
        targets = targets.to(device)

        batch_size = data.size(0)

        # with torch.cuda.amp.autocast():
        with amp.autocast(enabled=True):
            outputs = model(data)
            # scores = torch.argmax(scores, dim=1).float()
            loss = loss_fn(outputs, targets)

        scaler.scale(loss).backward()

        scaler.step(optimizer)
        scaler.update()

        optimizer.zero_grad()

        if batch_idx % 1000 == 0 and batch_idx != 0:
            if scheduler is not None:
                scheduler.step()

        running_loss += (loss.item() * batch_size)
        dataset_size += batch_size

        epoch_loss = running_loss / dataset_size

        PREDS.append(torch.argmax(outputs, dim=1).cpu().detach().numpy())
        TARGETS.append(targets.cpu().detach().numpy())

        bar.set_postfix(Epoch=epoch, Train_Loss=epoch_loss, LR=optimizer.param_groups[0]['lr'])

    TARGETS = np.concatenate(TARGETS)
    PREDS = np.concatenate(PREDS)
    train_acc = accuracy_score(TARGETS, PREDS)

    # if scheduler is not None:
    #     scheduler.step()

    return epoch_loss, train_acc


@torch.no_grad()
def valid_one_epoch(model, loader, loss_fn, device, epoch):
    model.eval()

    dataset_size = 0
    running_loss = 0.0

    TARGETS = []
    PREDS = []

    bar = tqdm(enumerate(loader), total=len(loader))
    for step, (images, targets) in bar:
        images = images.to(device, dtype=torch.float)
        targets = targets.to(device, dtype=torch.long)

        batch_size = images.size(0)

        outputs = model(images)
        loss = loss_fn(outputs, targets)

        running_loss += (loss.item() * batch_size)
        dataset_size += batch_size

        epoch_loss = running_loss / dataset_size

        PREDS.append(torch.argmax(outputs, dim=1).cpu().detach().numpy())
        TARGETS.append(targets.cpu().detach().numpy())

        bar.set_postfix(Epoch=epoch, Valid_Loss=epoch_loss)

    TARGETS = np.concatenate(TARGETS)
    PREDS = np.concatenate(PREDS)
    val_acc = accuracy_score(TARGETS, PREDS)
    gc.collect()

    return epoch_loss, val_acc


def run_training(model, train_loader, valid_loader, optimizer, loss_fn, scheduler, experiment_path, device, num_epochs):
    # Path("logs").joinpath(config.MODEL_TYPE).joinpath("exp_" + str(exp_number)).mkdir(parents=True, exist_ok=True)

    if torch.cuda.is_available():
        print("[INFO] Using GPU: {}\n".format(torch.cuda.get_device_name()))

    start = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_epoch_acc = 0
    history = defaultdict(list)
    early_stop_patience = 5
    current_early_stop_patience = 0

    for epoch in range(1, num_epochs + 1):
        gc.collect()
        train_epoch_loss, train_epoch_acc = train_one_epoch(model, train_loader, loss_fn, optimizer, scheduler, device,
                                                            epoch)
        val_epoch_loss, val_epoch_acc = valid_one_epoch(model, valid_loader, loss_fn, device, epoch)

        history['Train Loss'].append(train_epoch_loss)
        history['Valid Loss'].append(val_epoch_loss)
        history['Train Acc'].append(train_epoch_acc)
        history['Valid Acc'].append(val_epoch_acc)

        print(f'Train Acc: {train_epoch_acc}')
        print(f'Valid Acc: {val_epoch_acc}')

        # deep copy the model
        if val_epoch_acc >= best_epoch_acc:
            best_epoch_acc = val_epoch_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            PATH = Path(experiment_path).joinpath(
                "e{:.0f}_val_loss_{:.4f}_val_acc_{:.4f}.pth".format(epoch, val_epoch_loss, val_epoch_acc))
            torch.save(model.state_dict(), PATH)

            end = time.time()
            time_elapsed = end - start
            print('Training complete in {:.0f}h {:.0f}m {:.0f}s'.format(
                time_elapsed // 3600, (time_elapsed % 3600) // 60, (time_elapsed % 3600) % 60))
            print("Best Accuracy: {:.4f}".format(best_epoch_acc))

            # load best model weights
            model.load_state_dict(best_model_wts)
        else:
            if current_early_stop_patience >= early_stop_patience:
                print(f"Early stop on epoch {epoch}")
                break
            else:
                current_early_stop_patience += 1

        print()

    end = time.time()
    time_elapsed = end - start
    print('Training complete in {:.0f}h {:.0f}m {:.0f}s'.format(
        time_elapsed // 3600, (time_elapsed % 3600) // 60, (time_elapsed % 3600) % 60))
    print("Best Accuracy: {:.4f}".format(best_epoch_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)

    return model, history


def draw_result(lst_iter, train_loss, val_loss, train_acc, val_acc):
    fig, axis = plt.subplots(nrows=1, ncols=2, figsize=(12, 6), sharey=True, sharex=True)

    axis[0].plot(lst_iter, train_loss, '-b', label='Training loss')
    axis[0].plot(lst_iter, val_loss, '-g', label='Validation loss')
    axis[0].set_title("Training and Validation loss")
    axis[0].legend()

    axis[1].plot(lst_iter, train_acc, '-b', label='Training acc')
    axis[1].plot(lst_iter, val_acc, '-g', label='Validation acc')
    axis[1].set_title("Training and Validation acc")
    axis[1].legend()

    # fig.tight_layout()

    # plt.plot(lst_iter, train_loss, '-b', label='Training loss')
    # plt.plot(lst_iter, val_loss, '-g', label='Validation loss')
    #
    # plt.title('Training and Validation loss')
    # plt.xlabel('Epochs')
    # plt.ylabel('Loss')
    # plt.legend()

    # save image
    plt.savefig("result.png")  # should before show method

    # show
    plt.show()


def get_last_exp_number(model_name):
    folders = [x[0] for x in os.walk(os.path.join("logs", model_name))][1:]
    folders = [x.split("/")[-1] for x in folders]
    folders_exp = []
    for f in folders:
        if "exp" in f:
            folders_exp.append(f)

    if not folders_exp:
        return 0
    else:
        return max([int(x.split("_")[1]) for x in folders_exp]) + 1


def create_and_get_experiment_path() -> str:
    experiment_name = config.MODEL_TYPE
    experiment_number = get_last_exp_number(experiment_name)

    experiment_path = "logs" + os.sep + experiment_name + os.sep + "exp_" + str(experiment_number)

    Path(experiment_path).mkdir(parents=True, exist_ok=True)

    return experiment_path


def main():
    seed_everything(config.SEED)

    experiment_path = create_and_get_experiment_path()

    need_to_prepare_dataset = True
    if need_to_prepare_dataset:
        train_val_test_split(input_data_dir="data/dataset", output_data_dir="data/train_val_test_split")

    train_loader, val_loader, test_loader, dataset_train, dataset_val, dataset_test, label_encoder = dataset.get_loaders(
        config.IMAGE_SIZE,
        config.BATCH_SIZE,
        config.MEAN,
        config.STD,
        config.NUM_WORKERS)

    model = get_model(config.MODEL_TYPE, config.NUM_CLASSES)
    model = model.to(config.DEVICE)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    scheduler = fetch_scheduler(optimizer)

    model, history = run_training(model, train_loader, val_loader, optimizer, loss_fn, scheduler, experiment_path,
                                  device=config.DEVICE,
                                  num_epochs=config.NUM_EPOCHS)

    draw_result(range(config.NUM_EPOCHS), history['Train Loss'], history['Valid Loss'], history['Train Acc'],
                history['Valid Acc'])

    print(history)

    # if config.LOAD_MODEL and config.CHECKPOINT_FILE in os.listdir():
    #     load_checkpoint(torch.load(config.CHECKPOINT_FILE), model)

    # for epoch in range(config.NUM_EPOCHS):
    #     train_one_epoch(model, train_loader, loss_fn, optimizer, scheduler, config.DEVICE, epoch)
    # eval_one_epoch(val_loader, model, loss_fn, config.DEVICE, epoch)
    #     # check_accuracy(train_loader, model, loss_fn)
    #
    # if config.SAVE_MODEL:
    #     checkpoint = {"state_dict": model.state_dict(), "optimizer": optimizer.state_dict()}
    #     save_checkpoint(checkpoint, filename=config.CHECKPOINT_FILE)
    #
    # save_feature_vectors(model, train_loader, output_size=(1, 1), file="train_effb6")
    # save_feature_vectors(model, val_loader, output_size=(1, 1), file="test_effb6")


if __name__ == "__main__":
    main()
