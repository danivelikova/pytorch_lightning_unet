from pathlib import Path
import torch
import torchvision
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import configargparse
from pytorch_lightning.loggers import WandbLogger
from utils.plotter import Plotter
from utils.utils import (
    argparse_summary,
    get_class_by_path,
)
from pytorch_lightning import seed_everything
from utils.configargparse_arguments import build_configargparser
from datetime import datetime
import wandb
import os
import sys
import numpy as np
try:
    from utils.plx_logger import PolyaxonLogger
except ImportError:
    assert 'Import Error'

CROSS_VAL = False

polyaxon_folder = polyaxon_data_path + '1CT' \
    if CROSS_VAL is False else polyaxon_data_path + ''


def train(hparams, ModuleClass, ModelClass, DatasetClass, loggers, train_dataloader, val_loaders):
    model = ModelClass(hparams=hparams)
    module = ModuleClass(hparams, model, None)

    metric = "val_loss"
    checkpoint_callback = ModelCheckpoint(
        dirpath=f"{hparams.output_path}/checkpoints/",
        save_top_k=hparams.save_top_k,
        # save_last=True,
        verbose=True,
        monitor=metric,
        mode='min',
        prefix=hparams.name,
        filename=f'{{epoch}}-{{{metric}:.5f}}'
    )
    early_stop_callback = EarlyStopping(
        monitor=metric,
        min_delta=0.00,
        verbose=True,
        patience=3,
        mode='min')

    plotter = Plotter(hparams.current_train_fold)
    # ------------------------
    # 3 INIT TRAINER
    # ------------------------
    trainer = pl.Trainer(
        # gpus=hparams.gpus,
        logger=loggers,
        # fast_dev_run=hparams.fast_dev_run,
        min_epochs=hparams.min_epochs,
        max_epochs=hparams.max_epochs,
        checkpoint_callback=True,
        resume_from_checkpoint=hparams.resume_from_checkpoint,
        callbacks=[early_stop_callback, checkpoint_callback, plotter],
        weights_summary='full',
        # auto_lr_find=True,
        num_sanity_val_steps=hparams.num_sanity_val_steps,
        log_every_n_steps=hparams.log_every_n_steps,
        # check_val_every_n_epoch=1,  # how often the model will be validated
        limit_train_batches=hparams.limit_train,  # How much of training dataset to check (if 1 check all)
        limit_val_batches=hparams.limit_val,
        # limit_test_batches=LIMIT_TEST,
        # auto_scale_batch_size=True,   # search for optimal batch size
        # automatic_optimization=True,
        profiler="simple"
    )
    print('MIN EPOCHS: ', trainer.min_epochs)
    print('MAX EPOCHS: ', trainer.max_epochs)

    # wandb_logger.watch(module, log="all", log_freq=100)  # plots the gradients
    print(f'---------------TRAIN FIT--------------\n'
          f'VAL DISABLED?: {trainer.disable_validation} \n'
          f'VAL ENABLED?: {trainer.enable_validation}')
    seed_everything(seed=0)
    trainer.fit(module, train_dataloader=train_dataloader, val_dataloaders=val_loaders)
    print('---------------TEST--------------')
        # trainer.test(test_dataloaders=test_loaders)

    return trainer.checkpoint_callback.best_model_score  # best_model_loss


# LOAD MODULE
def load_module(hparams):
    module_path = f"modules.{hparams.module}"
    ModuleClass = get_class_by_path(module_path)
    # parser = ModuleClass.add_module_specific_args(parser)
    # ------------------------
    # LOAD MODEL
    # ------------------------
    model_path = f"models.{hparams.model}"
    ModelClass = get_class_by_path(model_path)
    # parser = OuterModelClass.add_model_specific_args(parser)

    return ModuleClass, ModelClass


# # LOAD DATASET
def load_dataset(hparams):
    dataset_path = f"datasets.{hparams.dataset}"
    DatasetClass = get_class_by_path(dataset_path)
    # parser = DatasetClass.add_dataset_specific_args(parser)
    return DatasetClass


if __name__ == "__main__":
    # ------------------------
    # TRAINING ARGUMENTS
    # ------------------------
    # these are project-wide arguments
    root_dir = Path(__file__).parent
    parser = configargparse.ArgParser(
        config_file_parser_class=configargparse.YAMLConfigFileParser)
    parser.add('-c', is_config_file=True, help='config file path')
    parser, hparams = build_configargparser(parser)

    print(f'************************************************************************************* \n'
          f'COMMENT: {COMMENT + str(hparams.nr_train_folds)} \n'
          f'************************************************************************************* \n'
          f'PL VERSION: {pl.__version__} \n'
          f'PYTHON VERSION: {sys.version} \n '
          # f'WANDB VERSION: {wandb.__version__} \n '
          f'TORCH VERSION: {torch.__version__} \n '
          f'TORCHVISION VERSION: {torchvision.__version__}')

    ModuleClass, ModelClass,  = load_module(hparams)
    parser = ModuleClass.add_module_specific_args(parser)
    parser = ModelClass.add_model_specific_args(parser)
    DatasetClass = load_dataset(hparams)
    parser = DatasetClass.add_dataset_specific_args(parser)

    hparams = parser.parse_args()
    # setup logging
    exp_name = (
            hparams.outer_module.split(".")[-1]
            + "_"
            + hparams.dataset.split(".")[-1]
            + "_"
            + hparams.outer_model.replace(".", "_")
    )
    print(f'This will run on polyaxon: {str(hparams.on_polyaxon)}')

    # hparams.device = torch.device('cuda' if hparams.on_polyaxon else 'cpu')
    hparams.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('torch.cuda.is_available(): ', torch.cuda.is_available())
    print('device: ', hparams.device)

    if hparams.on_polyaxon:
        from polyaxon_client.tracking import Experiment, get_data_paths, get_outputs_path

        hparams.data_root = get_data_paths()['data1'] + polyaxon_folder
        hparams.output_path = get_outputs_path()
        poly_experiment_info = Experiment.get_experiment_info()
        poly_experiment_nr = poly_experiment_info['experiment_name'].split(".")[-1]
        hparams.name = poly_experiment_nr + "_" + exp_name
        print(f'get_outputs_path: {get_outputs_path()} \n '
              f'experiment_info: {poly_experiment_info} \n experiment_name: {poly_experiment_nr}')

    else:
        date_str = datetime.now().strftime("%y%m%d-%H%M%S_")
        hparams.name = 'local_' + date_str + exp_name
        # hparams.output_path = Path(hparams.output_path).absolute() / hparams.name

    wandb_logger = WandbLogger(name=hparams.name,
                               project=f"aortaSegm-{hparams.outer_model.split('.')[-1]}-{hparams.inner_module.split('.')[-1]}")
    # wandb.init(project=f"aortaSegm-{hparams.outer_model.split('.')[-1]}-{hparams.inner_module.split('.')[-1]}")

    argparse_summary(hparams, parser)
    loggers = wandb_logger

    # ---------------------
    # LOAD DATA LOADERS
    # ---------------------
    best_model_loss_list = []
    if CROSS_VAL:
        list_dirs = sorted(os.listdir(hparams.data_root))
        for idx, fold in enumerate(list_dirs):
            # merge datasets for cross val
            if idx + hparams.nr_train_folds <= len(list_dirs):
                train_folds = list_dirs[idx:idx + hparams.nr_train_folds]
            else:
                train_folds = list_dirs[idx:] + list_dirs[:hparams.nr_train_folds - len(list_dirs) + idx]

            hparams.current_train_fold = ''.join(train_folds)
            # list_dirs = [hparams.data_root + '/' + d for d in sorted(os.listdir(hparams.data_root))]
            train_dataloaders = torch.utils.data.DataLoader(
                DatasetClass(hparams, train_folds), batch_size=hparams.batch_size, shuffle=True)

            rest_folds = [n for n in list_dirs if n not in train_folds]
            val_dataloaders = torch.utils.data.DataLoader(
                DatasetClass(hparams, rest_folds), batch_size=hparams.batch_size, shuffle=False)

            print('-------------------------- RUN TRAINING -----------------')
            print(f'------------ train_dataloaders: {train_folds}')
            print(f'------------ val_dataloaders: {rest_folds}')

            best_model_loss = train(hparams, ModuleClass, ModelClass, DatasetClass, loggers,
                                    train_dataloaders, val_dataloaders)
            best_model_loss_list.append(best_model_loss)

        print('-------------------------------------------------------------------------------------------------------')
        print(f'-------------------------- END CROSS VAL LOOP WITH NR_FOLDS: {hparams.nr_train_folds}-----------------')
        print(f'-------------------- list best losses: {best_model_loss_list}, \n ')
        print(f'-------------------- mean: {np.mean(best_model_loss_list):.4f}, \n ')
        print(f'-------------------- std: {np.std(best_model_loss_list):.4f}')
        print(f'------------------------------------------------------------------------------------------------------')
        del best_model_loss_list[:]

    else:
        train_loader = torch.utils.data.DataLoader(
            DatasetClass(hparams, "train/"), batch_size=hparams.batch_size, shuffle=True)
        val_loader = torch.utils.data.DataLoader(
            DatasetClass(hparams, "val/"), batch_size=hparams.batch_size, shuffle=False)
        # test_loader = torch.utils.data.DataLoader(
        #     DatasetClass(hparams, "test/"), batch_size=hparams.batch_size, shuffle=False)

        # ---------------------
        # RUN TRAINING
        # ---------------------
        train(hparams, ModuleClass, ModelClass, DatasetClass, loggers, train_loader, val_loader)