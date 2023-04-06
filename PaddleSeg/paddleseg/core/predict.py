# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import math

import cv2
import numpy as np
import paddle

from paddleseg import utils
from paddleseg.core import infer
from paddleseg.utils import logger, progbar, visualize

import numpy as np

def testIntensity(src):
    margin = 10
    controlToTest = 100
    scan = 10
    refScan = 10
    lineWidth = 6
    src = cv2.cvtColor(src, cv2.COLOR_RGB2GRAY)
    img = cv2.rotate(src, cv2.ROTATE_90_COUNTERCLOCKWISE)
    
    h,w = img.shape
    
    img = img[margin:h-margin, margin:w-margin]
    mean = np.mean(img,axis=0)
    mean = 255 - mean

    controlToTest = int(w/3)
    
    LFAControl = mean[0:int(w/2)]
    if(len(LFAControl)==0):
        return 0
    ControlLineLoc = np.argmax(LFAControl)
    
    TestLineLoc = ControlLineLoc + controlToTest
    TestLineLocation = TestLineLoc-scan + np.argmax(mean[TestLineLoc-scan:TestLineLoc+scan])
    
    ReferenceLocation = int((ControlLineLoc + TestLineLocation)/2)
    Reference = np.mean(mean[ReferenceLocation-refScan:ReferenceLocation+refScan])
    
    LFACorrected = mean-Reference
    LFACorrected[LFACorrected < 0] = 0
    
    ControlLine = np.amax(LFACorrected[0:int(w/2)])
    
    TestLine = np.sum(LFACorrected[TestLineLocation-lineWidth:TestLineLocation+lineWidth])
    TestLineAvg = np.mean(LFACorrected[TestLineLocation-lineWidth:TestLineLocation+lineWidth])
    return TestLine

def contour_area(contours):
     
    # create an empty list
    # cnt_rect = []
    min = 1.0
    x_o = 0
    y_o = 0 
    w_o = 0
    h_o = 0
     
    # loop through all the contours
    for i in range(0,len(contours),1):
        # for each contour, use OpenCV to calculate the area of the contour
        cnt = contours[i]
        x,y,w,h = cv2.boundingRect(cnt)
        rect_area = w*h
        cnt_area = cv2.contourArea(cnt)
        # cnt_rect.append((rect_area-cnt_area)/float(rect_area))
        rect_frac = (rect_area-cnt_area)/float(rect_area)
        # print(rect_frac)
        if rect_frac < min:
            min = rect_frac
            x_o,y_o,w_o,h_o = x,y,w,h
 
    # Sort our list of contour areas in descending order
    # list.sort(cnt_rect)
    return x_o,y_o,w_o,h_o

def draw_bounding_box(contours, image, number_of_boxes=1):
    # # Call our function to get the list of contour areas
    # cnt_area = contour_area(contours)
 
    # # Loop through each contour of our image
    # for i in range(0,len(contours),1):
    #     cnt = contours[i]
 
    #     # Only draw the the largest number of boxes
    #     if (cv2.contourArea(cnt) >= cnt_area[number_of_boxes-1]):
             
    #         # Use OpenCV boundingRect function to get the details of the contour
    #         x,y,w,h = cv2.boundingRect(cnt)
             
    #         # Draw the bounding box
    #         out=image[y:y+h,x:x+w].copy()

    x,y,w,h = contour_area(contours)
    out=image[y:y+h,x:x+w].copy()
 
    return out


def mkdir(path):
    sub_dir = os.path.dirname(path)
    if not os.path.exists(sub_dir):
        os.makedirs(sub_dir)


def partition_list(arr, m):
    """split the list 'arr' into m pieces"""
    n = int(math.ceil(len(arr) / float(m)))
    return [arr[i:i + n] for i in range(0, len(arr), n)]


def preprocess(im_path, transforms):
    data = {}
    data['img'] = im_path
    data = transforms(data)
    data['img'] = data['img'][np.newaxis, ...]
    data['img'] = paddle.to_tensor(data['img'])
    return data


def predict(model,
            model_path,
            transforms,
            image_list,
            image_dir=None,
            save_dir='output',
            aug_pred=False,
            scales=1.0,
            flip_horizontal=True,
            flip_vertical=False,
            is_slide=False,
            stride=None,
            crop_size=None,
            custom_color=None):
    """
    predict and visualize the image_list.

    Args:
        model (nn.Layer): Used to predict for input image.
        model_path (str): The path of pretrained model.
        transforms (transform.Compose): Preprocess for input image.
        image_list (list): A list of image path to be predicted.
        image_dir (str, optional): The root directory of the images predicted. Default: None.
        save_dir (str, optional): The directory to save the visualized results. Default: 'output'.
        aug_pred (bool, optional): Whether to use mulit-scales and flip augment for predition. Default: False.
        scales (list|float, optional): Scales for augment. It is valid when `aug_pred` is True. Default: 1.0.
        flip_horizontal (bool, optional): Whether to use flip horizontally augment. It is valid when `aug_pred` is True. Default: True.
        flip_vertical (bool, optional): Whether to use flip vertically augment. It is valid when `aug_pred` is True. Default: False.
        is_slide (bool, optional): Whether to predict by sliding window. Default: False.
        stride (tuple|list, optional): The stride of sliding window, the first is width and the second is height.
            It should be provided when `is_slide` is True.
        crop_size (tuple|list, optional):  The crop size of sliding window, the first is width and the second is height.
            It should be provided when `is_slide` is True.
        custom_color (list, optional): Save images with a custom color map. Default: None, use paddleseg's default color map.

    """
    utils.utils.load_entire_model(model, model_path)
    model.eval()
    nranks = paddle.distributed.get_world_size()
    local_rank = paddle.distributed.get_rank()
    if nranks > 1:
        img_lists = partition_list(image_list, nranks)
    else:
        img_lists = [image_list]

    added_saved_dir = os.path.join(save_dir, 'added_prediction')
    pred_saved_dir = os.path.join(save_dir, 'pseudo_color_prediction')
    cropped_saved_dir = os.path.join(save_dir, 'cropped')

    out_int = np.zeros(len(img_lists[local_rank]))

    logger.info("Start to predict...")
    progbar_pred = progbar.Progbar(target=len(img_lists[0]), verbose=1)
    color_map = visualize.get_color_map_list(256, custom_color=custom_color)
    with paddle.no_grad():
        for i, im_path in enumerate(img_lists[local_rank]):
            data = preprocess(im_path, transforms)

            if aug_pred:
                pred, _ = infer.aug_inference(
                    model,
                    data['img'],
                    trans_info=data['trans_info'],
                    scales=scales,
                    flip_horizontal=flip_horizontal,
                    flip_vertical=flip_vertical,
                    is_slide=is_slide,
                    stride=stride,
                    crop_size=crop_size)
            else:
                pred, _ = infer.inference(
                    model,
                    data['img'],
                    trans_info=data['trans_info'],
                    is_slide=is_slide,
                    stride=stride,
                    crop_size=crop_size)
            pred = paddle.squeeze(pred)
            pred = pred.numpy().astype('uint8')

            # get the saved name
            if image_dir is not None:
                im_file = im_path.replace(image_dir, '')
            else:
                im_file = os.path.basename(im_path)
            if im_file[0] == '/' or im_file[0] == '\\':
                im_file = im_file[1:]

            # save added image
            added_image = utils.visualize.visualize(
                im_path, pred, color_map, weight=0.6)
            added_image_path = os.path.join(added_saved_dir, im_file)
            mkdir(added_image_path)
            cv2.imwrite(added_image_path, added_image)

            # save pseudo color prediction
            pred_mask = utils.visualize.get_pseudo_color_map(pred, color_map)
            pred_saved_path = os.path.join(
                pred_saved_dir, os.path.splitext(im_file)[0] + ".png")
            mkdir(pred_saved_path)
            pred_mask.save(pred_saved_path)

            img = cv2.imread(im_path)

            contours,hierarchy = cv2.findContours(pred, 1, 2)
            cropped = draw_bounding_box(contours,img)
            cropped_saved_path = os.path.join(
                cropped_saved_dir, os.path.splitext(im_file)[0] + "_cropped.png")
            mkdir(cropped_saved_path)
            cv2.imwrite(cropped_saved_path, cropped)

            intensity = testIntensity(cropped)
            
            out_int[i] = intensity

            print("Intensity: " + str(intensity))

            progbar_pred.update(i + 1)
    return out_int
