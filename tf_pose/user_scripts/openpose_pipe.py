import argparse
import logging
import sys
import time
import os
import csv

#sys.path.insert(0, '../tf_pose')

from tf_pose import common
import pickle
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh

logger = logging.getLogger('TfPoseEstimator')
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)


def load_model_graph(model):
    estimator = TfPoseEstimator(get_graph_path(model)) 
    return estimator

def initial_preprocessing(estimator, image_file, resize='0x0'):
    w, h = model_wh(resize) 

    w, h = (432, 368) if (w, h) == (0, 0) else (w, h)
    
    estimator.initialize_hyperparams(target_size=(w, h))

    image = common.read_imgfile(image_file, None, None)
    if image is None:
        logger.error('Image can not be read, path=%s' % image)
        sys.exit(-1)
    
    return estimator, image, (w, h)


def get_pose_estimation(estimator, image, resolution, resize_ratio=4.0, show_background=False):
    w, h = resolution
    humans_estimator = estimator.inference(image, resize_to_default=(w > 0 and h > 0), upsample_size=resize_ratio)
    
    if not show_background:
        image = np.zeros((w, h, 3), np.uint8)
        image[:,:] = (255,255,255) 

    # Create blank image with pose estimator's skeleton
    image = TfPoseEstimator.draw_humans(image, humans_estimator, imgcopy=False) 

    return image, humans_estimator


def get_estimator_joints(humans_estimator, resolution):
    w, h = resolution
    joints = TfPoseEstimator.get_joints(humans_estimator, w, h)

    return joints


def save_pose_estimation_image(image, path, debug=False):
    if debug:
        print('\nSave image at ' + path + '\n')
    cv2.imwrite(path, image)


def show_image(image):
    plt.imshow(image)
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='tf-pose-estimation run')
    parser.add_argument('--image', type=str, default='../images/p1.jpg')
    parser.add_argument('--model', type=str, default='mobilenet_thin', help='cmu / mobilenet_thin')

    parser.add_argument('--resize', type=str, default='0x0',
                        help='if provided, resize images before they are processed. default=0x0, Recommends : 432x368 or 656x368 or 1312x736 ')
    parser.add_argument('--resize-out-ratio', type=float, default=4.0,
                        help='if provided, resize heatmaps before they are post-processed. default=1.0')
    args = parser.parse_args()

    # Get image and estimator object, load model graphs
    estimator, image, resolution = initial_preprocessing(args.model, args.image, args.resize)

    # Get skeleton's image and humans estimator object (contains heatmaps, vector fields, etc)
    pose_est_img, humans_estim = get_pose_estimation(estimator, image, resolution, args.resize_out_ratio)    

    # Get skeleton's joints coords
    joints = get_estimator_joints(humans_estim, resolution)

    # Show and save image
    path = os.path.join(os.getcwd(), 'image.png')
    save_pose_estimation_image(pose_est_img, path, debug=True)
    show_image(pose_est_img)


if __name__ == '__main__':
    main()
 
