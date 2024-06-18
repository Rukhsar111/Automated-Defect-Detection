"""Anomalib Training Script.

This script reads the name of the model or config file from command
line, train/test the anomaly model to get quantitative and qualitative
results.
"""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
import warnings
from argparse import ArgumentParser, Namespace

from pytorch_lightning import Trainer, seed_everything

from anomalib.config import get_configurable_parameters
from anomalib.data import get_datamodule
from anomalib.data.utils import TestSplitMode
from anomalib.models import get_model
from anomalib.utils.callbacks import LoadModelCallback, get_callbacks
from anomalib.utils.loggers import configure_logger, get_experiment_logger

logger = logging.getLogger("anomalib")

import numpy as np
import cv2
import os
import yaml

import os,shutil

config_file="/home/Automated-Defect-Detection/configs/default.yaml"

# Load the default configuration parametres.
def load_configuration(config_file):
    
    with open(f'{config_file}', 'r+') as yaml_file:
        configurations=yaml.safe_load(yaml_file)
        # print(type(configurations))
    return configurations


# Upate the default configurations with the custom parameters.
def update_configurations(data_root_path,folder_name,name_normal_dir,name_abnormal_dir,task):
    
    #load the config file that needs to be modified.
    with open(f'{config_file}') as yaml_file:
        data=yaml.safe_load(yaml_file)
      
    # Apply changes in the configuration  file.
    with open(f'{config_file}' ,'w') as file:
        data['dataset'] ['path']= data_root_path
        data['dataset'] ['name']= folder_name
        data['dataset'] ['normal_dir']= name_normal_dir
        data['dataset'] ['abnormal_dir']=name_abnormal_dir
        data['dataset'] ['task']=task
        yaml.safe_dump(data , file)


## starts training on the provided data.
def train_model(config_file_path):
    """Train an Defect Detection model """

   

    config = get_configurable_parameters(model_name="padim", config_path=config_file_path)
   

    # if config.project.get("seed") is not None:
    #     seed_everything(config.project.seed)

    datamodule = get_datamodule(config)
    model = get_model(config)
    experiment_logger = get_experiment_logger(config)
    callbacks = get_callbacks(config)

    trainer = Trainer(**config.trainer, logger=experiment_logger, callbacks=callbacks)
    logger.info("Training the model.")
    trainer.fit(model=model, datamodule=datamodule)

    logger.info("Loading the best model weights.")
    load_model_callback = LoadModelCallback(weights_path=trainer.checkpoint_callback.best_model_path)
    trainer.callbacks.insert(0, load_model_callback)  # pylint: disable=no-member

    if config.dataset.test_split_mode == TestSplitMode.NONE:
        logger.info("No test set provided. Skipping test stage.")
    else:
        logger.info("Testing the model.")
        trainer.test(model=model, datamodule=datamodule)


if __name__ == "__main__":

    data_root_path="data/cast_defect"
    name="Bearing"   ##name of the folder or the product name
    name_normal_dir="cast_ok"
    name_abnormal_dir="cast_def"
    task="classification"

    ##update the config with your custom dataset 
    updated_config=update_configurations(data_root_path,name,name_normal_dir,name_abnormal_dir,task)

    #Train the Model on custom data using the updated config file.
    train_model(config_file)




