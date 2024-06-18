"""Anomalib Torch Inferencer Script.

This script performs torch inference by reading model weights
from command line, and show the visualization results.
"""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import warnings
from argparse import ArgumentParser, Namespace
from pathlib import Path
import cv2

import matplotlib.pyplot as plt
import torch

from anomalib.data.utils import generate_output_image_filename, get_image_filenames, read_image
from anomalib.deploy import TorchInferencer
from anomalib.post_processing import Visualizer

from torch import as_tensor
from torchvision.transforms.v2.functional import to_dtype, to_image
from anomalib.post_processing.post_process import (
    add_anomalous_label,
    add_normal_label)


visualization_mode='full'

show_output=True

def display_res(output):
    plt.imshow(output)
    plt.axis('off')
    plt.show()


def show_results(image, heat_map ,anomaly_map, segmentations , pred_label, pred_score, image_classified):
    print(f"pred_label, pred_score: {pred_label, pred_score}")
    
    
    
    # Show the image in case the flag is set by the user.
    if show_output:
        
        plt.subplot(2,3,1)
        plt.title("Original")
        plt.imshow(image)
        plt.axis('off')

        plt.subplot(2,3,2)
        plt.title("HeatMap")
        plt.imshow(heat_map)
        plt.axis('off')
        
        plt.subplot(2,3,3)
        plt.title("Anomaly_map")
        plt.imshow(anomaly_map)
        plt.axis('off')
    
        plt.subplot(2,3,4)
        plt.title("Segmented")
        plt.imshow(segmentations)
        plt.axis('off')

        plt.subplot(2,3,5)
        plt.title("Predicted")
        plt.imshow(image_classified)
        plt.axis('off')


        plt.show()

        plt.savefig("result.png")


def inference_engine() :
    """Infer predictions.

    Show/save the output if path is to an image. If the path is a directory, go over each image in the directory.

    Args:
        args (Namespace): The arguments from the command line.
    """
    torch.set_grad_enabled(False)
    
    filenames = get_image_filenames(data_dir_path)
    for filename in filenames:
        image = read_image(filename)
        results = inferencer.predict(image=image)
        output = visualizer.visualize_image(results)
        org_img=results.image.copy()
        
        if out_path is None  and  visualization_mode is False:
            print("Neither output path is provided nor show flag is set. Inferencer will run but return nothing.")

        if isinstance(out_path, str):
            file_path = generate_output_image_filename(input_path=filename, output_path=out_path)
            visualizer.save(file_path=file_path, image=output)

        score=results.pred_score
        print(score)

        if results.pred_score>0.6:
                image_classified = add_anomalous_label(results.image, results.pred_score)
        elif results.pred_score<0.6:
            image_classified = add_normal_label(results.image, 1 - results.pred_score)
        
        
        
        #Visualise the output results
        # display_res(image_classified)

    
    return org_img, results.heat_map , results.anomaly_map ,results.segmentations, results.pred_label , results.pred_score , image_classified


if __name__ == "__main__":

    # provide the trained model path.
    path_to_model_weights="/home/Anomalib/weights.pt"

    #Data Dir path.
    data_dir_path="/home/Anomalob/test"
   
    #Output dir path in order to save the infrenced result.
    out_path="/home/Anomalib/out_reslts"

    
    # Load the model
    print("load model")
    inferencer = TorchInferencer(path=path_to_model_weights, device="cpu")

    #Initaite the Visualizer in order to visualize
    visualizer = Visualizer(mode=visualization_mode, task='classification')


    #Call the inference_engine method and perform inferencing.
    # inference_engine(data_dir_path,path_to_model_weights,out_path,visualization_mode, show_output,inferencer )
    image, heat_map, anomaly_map, segmentations, pred_label, pred_score , image_classified =inference_engine()

    show_results(image, heat_map, anomaly_map, segmentations, pred_label, pred_score ,image_classified)

    
