import numpy as np
# from skimage.io import imread, imshow, imsave
from mrcnn import model as modellib # proposed --> mask_rcnn_nucleus_5000.h5
# from mrcnn import model as modellib # maskrcnn_og --> mask_rcnn_nucleus_7989.h5 in /home/xuan/Desktop/Mask_RCNN-master/logs/nucleus20210318T1345/

from mrcnn.config import Config
from keras import backend as K
import scipy.io as sio 
import cv2
import glob
# from psutil import virtual_memory
import matplotlib.pyplot as plt
import os
import sys
import threading
# from queue import Queue
import multiprocessing
from tqdm import tqdm
import matplotlib.pyplot as plt
from skimage.morphology import dilation,disk, closing, square, opening, erosion
from skimage import measure
import time
import pandas as pd 
import pdb
import scipy.ndimage
import json
import csv  
# from pathlib import Path

def overlap_preposes(input_):
    result = input_[0]
    size   = input_[1]
    th = 0.5
    result = np.array(result)
    mask_overlap = np.zeros(size)
    delete = []
    for l in tqdm(range(result.shape[0])):
        mask_out = np.zeros(size).astype(np.int32)
        mask_out[result[l,1][0] : result[l,1][1], result[l,1][2] :result[l,1][3]] = result[l,2]
        mask_overlap = mask_overlap + mask_out * (l+1)
        if mask_overlap.max() > (l+1):
            pixel_number = result[int(l),2].sum()
            number = mask_overlap[np.where(mask_overlap > (l+1))]
            uni_number = np.unique(number)
            for uni in uni_number:
                pixel_number2 = result[int(uni-l-1-1),2].sum()
                overlap  = len(np.where(number == uni)[0])/pixel_number
                overlap2 = len(np.where(number == uni)[0])/pixel_number2
                if overlap < th:
                    if overlap2 > th:
                        mask_overlap[np.where(mask_overlap == uni)] = (l+1)
                        mask_overlap[np.where(mask_overlap == (uni - l - 1))]   = 0
                        delete.append(int(uni-l-1-1))
                    else:
                        mask_overlap[np.where(mask_overlap == uni)] = (l+1)
                if overlap > th:
                    if pixel_number2 >= pixel_number:
                        mask_overlap[np.where(mask_overlap == uni)] = (uni - l - 1)
                        mask_overlap[np.where(mask_overlap == (l+1))]    = 0
                        delete.append(l)
                    if pixel_number2 < pixel_number:
                        mask_overlap[np.where(mask_overlap == uni)] = (l+1)
                        mask_overlap[np.where(mask_overlap == (uni - l - 1))]   = 0
                        delete.append(int(uni-l-1-1))
                    if pixel_number2 < 50:
                        delete.append(int(uni-l-1-1))
                    
    if delete != []:
        result = np.delete(result,delete,0)
    return result.tolist()


def th_overlap_preposes(result,size):
    result_size = len(result)//8
    pool = multiprocessing.Pool(processes= 8)
    start = len(result)
    print('start = ' +  str(start))
    print(len(result[0::8]) + len(result[1::8]) + len(result[2::8]) +
          len(result[3::8]) +len(result[4::8]) +len(result[5::8]) +
          len(result[6::8]) +len(result[7::8]))
    tasks = [(result[0::8],size),
             (result[1::8],size),
             (result[2::8],size),
             (result[3::8],size),
             (result[4::8],size),
             (result[5::8],size),
             (result[6::8],size),
             (result[7::8],size),]
    result_mult = pool.map(overlap_preposes, tasks)
    result = result_mult[0] + result_mult[1] + \
                  result_mult[2] + result_mult[3] + \
                  result_mult[4] + result_mult[5] + \
                  result_mult[6] + result_mult[7]
    pool.close()
    pool.join()
    del result_mult
    end1 = len(result)
    print('end1 = ' + str(end1))
    result = overlap_preposes((result,size))
    end2 = len(result)
    print('end2 = ' + str(end2))
    return result

def overlap_pred(model,image,name,size=(512,512),overlap =(256,256)):
    masks_new  = []
    step_number = 0
    for i in range((image.shape[0]//overlap[0]) + 1):
        for j in range((image.shape[1]//overlap[1]) + 1):
            x_title = i*overlap[0]
            y_title = j*overlap[1]
            x_end   = x_title + size[0]
            y_end   = y_title + size[1]
            if y_end > image.shape[1]: #if y_end exceed the width of image
                y_end = image.shape[1] #choose the longest width of image
            if x_end > image.shape[0]:
                x_end = image.shape[0]
            pre_image = image[x_title:x_end,y_title:y_end,:] #crop the image 
            rangemask = np.zeros(size) #default = 512*512
            rangemask[0:pre_image.shape[0],0:pre_image.shape[1]] = np.ones(pre_image.shape[0:2]) #
            pre_image_pad = np.ones(size + (3,))*255
            pre_image_pad[0:pre_image.shape[0],0:pre_image.shape[1],:] = pre_image
            result = model.detect([pre_image_pad], verbose=0)[0] #predict cells in assigned area            
            
            # print(result)
            scores = result['scores']
            class_ids = result['class_ids']
            masks     = result['masks']
            rois      = result['rois']
            del result
            for num, c in enumerate(class_ids):
                mask = (masks[:,:,num] * rangemask).astype(np.bool) #cell mask (binary)
                x0 = rois[num,:][0] #coordinate of cells
                x1 = rois[num,:][2] 
                y0 = rois[num,:][1] 
                y1 = rois[num,:][3]
                if x1 > pre_image.shape[0]:
                    x1 = pre_image.shape[0] 
                    rois[num,:][2] = rois[num,:][2] - (rois[num,:][2] - pre_image.shape[0])
                if y1 > pre_image.shape[1]:
                    y1 = pre_image.shape[1]
                    rois[num,:][3] = rois[num,:][3] - (rois[num,:][3] - pre_image.shape[1])
                x0 = rois[num,:][0] + x_title 
                x1 = rois[num,:][2] + x_title
                y0 = rois[num,:][1] + y_title
                y1 = rois[num,:][3] + y_title
                if mask.max() != 0:
                    roi_new = [x0, x1, y0, y1]
                    masks_new.append([c, roi_new, mask[rois[num,:][0]:rois[num,:][2] , rois[num,:][1]:rois[num,:][3]], scores[num]])
                    # print('width:',x1-x0,'height:',y1-y0)
                    # print('Prob:',scores[num])
                # if sys.getsizeof(masks_new) > 500000:
                #     print('reset_mask_new' + str(step_number))
                #     masks_new = th_overlap_preposes(masks_new,image.shape[0:2])
                #     # np.save('/media/xuan/TOSHIBA EXT/KI-67_DL/from_TVGH/image_png/roi_step/' + name + str(step_number) + '.npy', masks_new)
                #     # np.save('/mnt/disk1/OCELOT/result/' + name + str(step_number) + '.npy', masks_new)
                #     np.save(npy_path + str(name) + str(step_number) + '.npy', masks_new)
                #     step_number = step_number + 1
                #     del masks_new
                #     masks_new = []
                #     gc.collect()
                    
            print(sys.getsizeof(masks_new)) ####
            del masks
    masks_new = th_overlap_preposes(masks_new,image.shape[0:2])
    # step_file = sorted(glob.glob('/media/xuan/TOSHIBA EXT/KI-67_DL/from_TVGH/image_png/roi_step/' + name + '*.npy'))
    # step_file = sorted(glob.glob('/mnt/disk1/OCELOT/result/' + name + '*.npy'))
    # step_file = sorted(glob.glob(npy_path + str(name) + '*.npy'))
    step_file =[]
    if step_file != []:
        for st in step_file:
            print(st) ####
            step_merge = np.load(st,allow_pickle=True).tolist()
            masks_new  = masks_new + step_merge
            del step_merge
            masks_new = th_overlap_preposes(masks_new,image.shape[0:2])
            os.remove(st)
    return masks_new

def model_cell_load(weight_path):
    config = NucleusInferenceConfig()
    model  = modellib.MaskRCNN(mode="inference", config=config,model_dir=weight_path)
    model.load_weights(weight_path, by_name=True)
    return model

class NucleusConfig(Config):
    NAME = "nucleus"
    IMAGES_PER_GPU = 1
    NUM_CLASSES = 1 + 2 
    DETECTION_MIN_CONFIDENCE = 0.5
    BACKBONE = "resnet50"
    IMAGE_RESIZE_MODE = "crop"
    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 512
    IMAGE_MIN_SCALE = 1.0
    RPN_ANCHOR_SCALES = (8, 16, 24, 32, 64)
    RPN_TRAIN_ANCHORS_PER_IMAGE = 4096
    POST_NMS_ROIS_TRAINING = 2048
    POST_NMS_ROIS_INFERENCE = 3072
    RPN_NMS_THRESHOLD = 0.7
    MEAN_PIXEL = np.array([214.35, 219.74, 224.46])
    USE_MINI_MASK = True
    MINI_MASK_SHAPE = (56, 56)
    TRAIN_ROIS_PER_IMAGE = 1024
    MAX_GT_INSTANCES = 1024
    DETECTION_MAX_INSTANCES = 2048
    ROI_POSITIVE_RATIO = 0.33
    DETECTION_NMS_THRESHOLD = 0.3
    DETECTION_MIN_CONFIDENCE = 0.7
    FPN_CLASSIF_FC_LAYERS_SIZE = 1024
    Data_Set_Nms_max  = 0.7
    Data_Set_Nms_min  = 0.3
class NucleusInferenceConfig(NucleusConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    IMAGE_RESIZE_MODE = "pad64"


class Model():
    """
    Parameters
    ----------
    metadata: Dict
        Dataset metadata in case you wish to compute statistics

    """
    def __init__(self, metadata):
        self.metadata = metadata

    # def __call__(self, cell_patch, tissue_patch, pair_id,model_weight):
    #     """This function detects the cells in the cell patch. Additionally
    #     the broader tissue context is provided. 

    #     NOTE: this implementation offers a dummy inference example. This must be
    #     updated by the participant.

    #     Parameters
    #     ----------
    #     cell_patch: np.ndarray[uint8]
    #         Cell patch with shape [1024, 1024, 3] with values from 0 - 255
    #     tissue_patch: np.ndarray[uint8] 
    #         Tissue patch with shape [1024, 1024, 3] with values from 0 - 255
    #     pair_id: str
    #         Identification number of the patch pair

    #     Returns
    #     -------
    #         List[tuple]: for each predicted cell we provide the tuple (x, y, cls, score)
    #     """
    #     # Getting the metadata corresponding to the patch pair ID
    #     meta_pair = self.metadata[pair_id]

    #     #############################################
    #     #### YOUR INFERENCE ALGORITHM GOES HERE #####
    #     #############################################

    #     # The following is a dummy cell detection algorithm
    #     prediction = np.copy(cell_patch[:, :, 2])
    #     prediction[(cell_patch[:, :, 2] <= 40)] = 1
    #     xs, ys = np.where(prediction.transpose()== 1)
    #     class_id = [1] * len(xs) # Type of cell
    #     probs = [1.0] * len(xs) # Confidence score
    #     print(xs)
    #     print(ys)
    #     print(class_id)
    #     print(probs)

    #     #############################################
    #     ####### RETURN RESULS PER SAMPLE ############
    #     #############################################

    #     # We need to return a list of tuples with 4 elements, i.e.:
    #     # - int: cell's x-coordinate in the cell patch
    #     # - int: cell's y-coordinate in the cell patch
    #     # - int: class id of the cell, either 1 (BC) or 2 (TC)
    #     # - float: confidence score of the predicted cell
    #     return list(zip(xs, ys, class_id, probs))


    def __call__(self, cell_patch, tissue_patch, pair_id,model_weight):
        """This function detects the cells in the cell patch. Additionally
        the broader tissue context is provided. 

        NOTE: this implementation offers a dummy inference example. This must be
        updated by the participant.

        Parameters
        ----------
        cell_patch: np.ndarray[uint8]
            Cell patch with shape [1024, 1024, 3] with values from 0 - 255
        tissue_patch: np.ndarray[uint8] 
            Tissue patch with shape [1024, 1024, 3] with values from 0 - 255
        pair_id: str
            Identification number of the patch pair

        Returns
        -------
            List[tuple]: for each predicted cell we provide the tuple (x, y, cls, score)
        """
        # Getting the metadata corresponding to the patch pair ID
        meta_pair = self.metadata[pair_id]
        print(meta_pair)

        #############################################
        #### YOUR INFERENCE ALGORITHM GOES HERE #####
        #############################################
        # npy_path = '/media/xuan/Transcend/OCELOT/n_result/'

        model = model_cell_load(model_weight)
        npy   = overlap_pred(model,cell_patch,pair_id,size=(512,512),overlap =(400,400))
        # print(len(r))
        # npy = np.save(npy_path + str(pair_id) + '.npy', r)
        # print(npy)

        xs, ys, class_id, probs = [], [], [], []
        print(xs)
        for num in range(len(npy)):
            x = int((npy[num][1][0]+npy[num][1][1])/2)
            y = int((npy[num][1][2]+npy[num][1][3])/2)
            cla = int(npy[num][0])
            probability = float(npy[num][3])

            xs.append(x)
            ys.append(y)
            class_id.append(cla)
            probs.append(probability)


        # # The following is a dummy cell detection algorithm
        # prediction = np.copy(cell_patch[:, :, 2])
        # prediction[(cell_patch[:, :, 2] <= 40)] = 1
        # xs, ys = np.where(prediction.transpose() == 1)
        # class_id = [1] * len(xs) # Type of cell
        # probs = [1.0] * len(xs) # Confidence score

        #############################################
        ####### RETURN RESULS PER SAMPLE ############
        #############################################

        # We need to return a list of tuples with 4 elements, i.e.:
        # - int: cell's x-coordinate in the cell patch
        # - int: cell's y-coordinate in the cell patch
        # - int: class id of the cell, either 1 (BC) or 2 (TC)
        # - float: confidence score of the predicted cell
        return list(zip(xs, ys, class_id, probs))