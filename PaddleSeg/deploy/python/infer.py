# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

import argparse
import codecs
import os
import sys
import cv2

LOCAL_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(LOCAL_PATH, '..', '..'))

import yaml
import numpy as np
from paddle.inference import create_predictor, PrecisionType
from paddle.inference import Config as PredictConfig

import paddleseg.transforms as T
from paddleseg.cvlibs import manager
from paddleseg.utils import get_sys_env, logger, get_image_list
from paddleseg.utils.visualize import get_pseudo_color_map


def parse_args(input):
    parser = argparse.ArgumentParser(description='Test')
    parser.add_argument(
        "--config",
        dest="cfg",
        help="The config file.",
        default=None,
        type=str,
        required=True)
    parser.add_argument(
        '--image_path',
        dest='image_path',
        help='The directory or path or file list of the images to be predicted.',
        type=str,
        default=None,
        required=True)
    parser.add_argument(
        '--batch_size',
        dest='batch_size',
        help='Mini batch size of one gpu or cpu.',
        type=int,
        default=1)
    parser.add_argument(
        '--save_dir',
        dest='save_dir',
        help='The directory for saving the predict result.',
        type=str,
        default='./output')
    parser.add_argument(
        '--device',
        choices=['cpu', 'gpu'],
        default="gpu",
        help="Select which device to inference, defaults to gpu.")

    parser.add_argument(
        '--use_trt',
        default=False,
        type=eval,
        choices=[True, False],
        help='Whether to use Nvidia TensorRT to accelerate prediction.')
    parser.add_argument(
        "--precision",
        default="fp32",
        type=str,
        choices=["fp32", "fp16", "int8"],
        help='The tensorrt precision.')
    parser.add_argument(
        '--min_subgraph_size',
        default=3,
        type=int,
        help='The min subgraph size in tensorrt prediction.')
    parser.add_argument(
        '--enable_auto_tune',
        default=False,
        type=eval,
        choices=[True, False],
        help='Whether to enable tuned dynamic shape. We uses some images to collect '
        'the dynamic shape for trt sub graph, which avoids setting dynamic shape manually.'
    )
    parser.add_argument(
        '--auto_tuned_shape_file',
        type=str,
        default="auto_tune_tmp.pbtxt",
        help='The temp file to save tuned dynamic shape.')

    parser.add_argument(
        '--cpu_threads',
        default=10,
        type=int,
        help='Number of threads to predict when using cpu.')
    parser.add_argument(
        '--enable_mkldnn',
        default=False,
        type=eval,
        choices=[True, False],
        help='Enable to use mkldnn to speed up when using cpu.')

    parser.add_argument(
        "--benchmark",
        type=eval,
        default=False,
        help="Whether to log some information about environment, model, configuration and performance."
    )
    parser.add_argument(
        "--model_name",
        default="",
        type=str,
        help='When `--benchmark` is True, the specified model name is displayed.'
    )

    parser.add_argument(
        '--with_argmax',
        dest='with_argmax',
        help='Perform argmax operation on the predict result.',
        action='store_true')
    parser.add_argument(
        '--print_detail',
        default=True,
        type=eval,
        choices=[True, False],
        help='Print GLOG information of Paddle Inference.')

    return parser.parse_args(input)


def use_auto_tune(args):
    return hasattr(PredictConfig, "collect_shape_range_info") \
        and hasattr(PredictConfig, "enable_tuned_tensorrt_dynamic_shape") \
        and args.device == "gpu" and args.use_trt and args.enable_auto_tune


def auto_tune(args, imgs, img_nums):
    """
    Use images to auto tune the dynamic shape for trt sub graph.
    The tuned shape saved in args.auto_tuned_shape_file.

    Args:
        args(dict): input args.
        imgs(str, list[str], numpy): the path for images or the origin images.
        img_nums(int): the nums of images used for auto tune.
    Returns:
        None
    """
    logger.info("Auto tune the dynamic shape for GPU TRT.")

    assert use_auto_tune(args), "Do not support auto_tune, which requires " \
        "device==gpu && use_trt==True && paddle >= 2.2"

    if not isinstance(imgs, (list, tuple)):
        imgs = [imgs]
    num = min(len(imgs), img_nums)

    cfg = DeployConfig(args.cfg)
    pred_cfg = PredictConfig(cfg.model, cfg.params)
    pred_cfg.enable_use_gpu(100, 0)
    if not args.print_detail:
        pred_cfg.disable_glog_info()
    pred_cfg.collect_shape_range_info(args.auto_tuned_shape_file)

    predictor = create_predictor(pred_cfg)
    input_names = predictor.get_input_names()
    input_handle = predictor.get_input_handle(input_names[0])

    for i in range(0, num):
        if isinstance(imgs[i], str):
            data = {'img': imgs[i]}
            data = np.array([cfg.transforms(data)['img']])
        else:
            data = imgs[i]
        input_handle.reshape(data.shape)
        input_handle.copy_from_cpu(data)
        try:
            predictor.run()
        except Exception as e:
            logger.info(str(e))
            logger.info(
                "Auto tune failed. Usually, the error is out of GPU memory "
                "for the model or image is too large. \n")
            del predictor
            if os.path.exists(args.auto_tuned_shape_file):
                os.remove(args.auto_tuned_shape_file)
            return

    logger.info("Auto tune success.\n")


class DeployConfig:
    def __init__(self, path):
        with codecs.open(path, 'r', 'utf-8') as file:
            self.dic = yaml.load(file, Loader=yaml.FullLoader)

        self._transforms = self.load_transforms(self.dic['Deploy'][
            'transforms'])
        self._dir = os.path.dirname(path)

    @property
    def transforms(self):
        return self._transforms

    @property
    def model(self):
        return os.path.join(self._dir, self.dic['Deploy']['model'])

    @property
    def params(self):
        return os.path.join(self._dir, self.dic['Deploy']['params'])

    @staticmethod
    def load_transforms(t_list):
        com = manager.TRANSFORMS
        transforms = []
        for t in t_list:
            ctype = t.pop('type')
            transforms.append(com[ctype](**t))

        return T.Compose(transforms)


class Predictor:
    def __init__(self, args):
        """
        Prepare for prediction.
        The usage and docs of paddle inference, please refer to
        https://paddleinference.paddlepaddle.org.cn/product_introduction/summary.html
        """
        self.args = args
        self.cfg = DeployConfig(args.cfg)

        self._init_base_config()

        if args.device == 'cpu':
            self._init_cpu_config()
        else:
            self._init_gpu_config()

        try:
            self.predictor = create_predictor(self.pred_cfg)
        except Exception as e:
            logger.info(str(e))
            logger.info(
                "If the above error is '(InvalidArgument) some trt inputs dynamic shape info not set, "
                "..., Expected all_dynamic_shape_set == true, ...', "
                "please set --enable_auto_tune=True to use auto_tune. \n")
            exit()

        if hasattr(args, 'benchmark') and args.benchmark:
            import auto_log
            pid = os.getpid()
            self.autolog = auto_log.AutoLogger(
                model_name=args.model_name,
                model_precision=args.precision,
                batch_size=args.batch_size,
                data_shape="dynamic",
                save_path=None,
                inference_config=self.pred_cfg,
                pids=pid,
                process_name=None,
                gpu_ids=0,
                time_keys=[
                    'preprocess_time', 'inference_time', 'postprocess_time'
                ],
                warmup=0,
                logger=logger)

    def _init_base_config(self):
        self.pred_cfg = PredictConfig(self.cfg.model, self.cfg.params)
        if not self.args.print_detail:
            self.pred_cfg.disable_glog_info()
        self.pred_cfg.enable_memory_optim()
        self.pred_cfg.switch_ir_optim(True)

    def _init_cpu_config(self):
        """
        Init the config for x86 cpu.
        """
        logger.info("Use CPU")
        self.pred_cfg.disable_gpu()
        if self.args.enable_mkldnn:
            logger.info("Use MKLDNN")
            # cache 10 different shapes for mkldnn
            self.pred_cfg.set_mkldnn_cache_capacity(10)
            self.pred_cfg.enable_mkldnn()
        self.pred_cfg.set_cpu_math_library_num_threads(self.args.cpu_threads)

    def _init_gpu_config(self):
        """
        Init the config for nvidia gpu.
        """
        logger.info("Use GPU")
        self.pred_cfg.enable_use_gpu(100, 0)
        precision_map = {
            "fp16": PrecisionType.Half,
            "fp32": PrecisionType.Float32,
            "int8": PrecisionType.Int8
        }
        precision_mode = precision_map[self.args.precision]

        if self.args.use_trt:
            logger.info("Use TRT")
            self.pred_cfg.enable_tensorrt_engine(
                workspace_size=1 << 30,
                max_batch_size=1,
                min_subgraph_size=self.args.min_subgraph_size,
                precision_mode=precision_mode,
                use_static=False,
                use_calib_mode=False)

            if use_auto_tune(self.args) and \
                os.path.exists(self.args.auto_tuned_shape_file):
                logger.info("Use auto tuned dynamic shape")
                allow_build_at_runtime = True
                self.pred_cfg.enable_tuned_tensorrt_dynamic_shape(
                    self.args.auto_tuned_shape_file, allow_build_at_runtime)
            else:
                logger.info("Use manual set dynamic shape")
                min_input_shape = {"x": [1, 3, 100, 100]}
                max_input_shape = {"x": [1, 3, 2000, 3000]}
                opt_input_shape = {"x": [1, 3, 512, 1024]}
                self.pred_cfg.set_trt_dynamic_shape_info(
                    min_input_shape, max_input_shape, opt_input_shape)

    def run(self, imgs_path):
        if not isinstance(imgs_path, (list, tuple)):
            imgs_path = [imgs_path]

        input_names = self.predictor.get_input_names()
        input_handle = self.predictor.get_input_handle(input_names[0])
        output_names = self.predictor.get_output_names()
        output_handle = self.predictor.get_output_handle(output_names[0])
        results = []
        args = self.args

        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)

        out_int = np.zeros(0)
        names = []

        for i in range(0, len(imgs_path), args.batch_size):
            # warm up
            if i == 0 and args.benchmark:
                for j in range(5):
                    data = np.array([
                        self._preprocess(img)
                        for img in imgs_path[0:args.batch_size]
                    ])
                    input_handle.reshape(data.shape)
                    input_handle.copy_from_cpu(data)
                    self.predictor.run()
                    results = output_handle.copy_to_cpu()
                    results = self._postprocess(results)

            # inference
            if args.benchmark:
                self.autolog.times.start()

            data = np.array([
                self._preprocess(p) for p in imgs_path[i:i + args.batch_size]
            ])
            input_handle.reshape(data.shape)
            input_handle.copy_from_cpu(data)

            if args.benchmark:
                self.autolog.times.stamp()

            self.predictor.run()

            results = output_handle.copy_to_cpu()
            if args.benchmark:
                self.autolog.times.stamp()

            results = self._postprocess(results)

            if args.benchmark:
                self.autolog.times.end(stamp=True)

            intensity, names2 = self._save_imgs(results, imgs_path[i:i + args.batch_size])

            if len(out_int) == 0:
                out_int = intensity
            else:
                out_int = np.append(out_int, intensity, axis=1)

            names += names2
        logger.info("Finish")
        return out_int, names

    def _preprocess(self, img):
        data = {}
        data['img'] = img
        return self.cfg.transforms(data)['img']

    def _postprocess(self, results):
        if self.args.with_argmax:
            results = np.argmax(results, axis=1)
        return results

    def _save_imgs(self, results, imgs_path):
        out_int = np.zeros((2,results.shape[0]))
        names = []
        for i in range(results.shape[0]):
            result = get_pseudo_color_map(results[i])
            basename = os.path.basename(imgs_path[i])
            basename, _ = os.path.splitext(basename)
            basename2 = f'{basename}.png'
            result.save(os.path.join(self.args.save_dir, basename2))

            img = cv2.imread(imgs_path[i])

            cropped_saved_dir = os.path.join(self.args.save_dir, 'cropped')

            contours,hierarchy = cv2.findContours(results[i].astype('uint8'), 1, 2)
            result, cropped, cntArea = draw_bounding_box(contours,img)
            if not result:
                out_int[0,i] = 0
                out_int[1,i] = 0
                names += basename
                continue

            cropped_saved_path = os.path.join(
                cropped_saved_dir, basename + "_cropped.png")
            mkdir(cropped_saved_path)
            cv2.imwrite(cropped_saved_path, cropped)

            intensity = testIntensity(cropped)
            
            out_int[0,i] = intensity
            out_int[1,i] = cntArea

            names += basename

            print("Intensity: " + str(intensity))
        return out_int, names

def mkdir(path):
    sub_dir = os.path.dirname(path)
    if not os.path.exists(sub_dir):
        os.makedirs(sub_dir)

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
    
    h,w = img.shape
    
    mean = np.mean(img,axis=0)
    mean = 255 - mean

    controlToTest = int(w/3)
    
    if controlToTest < 100:
        controlToTest = 100
    
    left=0
    right=int(w/2)
    
    LFAControl = mean[left:right]
    if(len(LFAControl)==0):
        return 0
    ControlLineLoc = np.argmax(LFAControl)
    
    
    while(ControlLineLoc == left or ControlLineLoc == right):
        if ControlLineLoc == left:
            left += margin
            LFAControl = mean[left:right]
        else:
            right -= margin
            LFAControl = mean[left:right]
        ControlLineLoc = left + np.argmax(LFAControl)
        
    TestLineLoc = ControlLineLoc + controlToTest

    if (TestLineLoc + scan >= len(mean)):
        scan = len(mean) - TestLineLoc

    try:
        TestLineLocation = TestLineLoc-scan + np.argmax(mean[TestLineLoc-scan:TestLineLoc+scan])
    except ValueError:
        return 0
    
    ReferenceLocation = int((ControlLineLoc + TestLineLocation)/2)
    Reference = np.mean(mean[ReferenceLocation-refScan:ReferenceLocation+refScan])
    
    LFACorrected = mean-Reference
    LFACorrected[LFACorrected < 0] = 0
    
    ControlLine = np.sum(LFACorrected[ControlLineLoc-lineWidth:ControlLineLoc+lineWidth])
    
    TestLine = np.sum(LFACorrected[TestLineLocation-lineWidth:TestLineLocation+lineWidth])
#     TestLineAvg = np.mean(LFACorrected[TestLineLocation-lineWidth:TestLineLocation+lineWidth]) 
    
    if ControlLine == 0:
        return 0
    
    return TestLine/ControlLine

def contour_area(contours):
     
    # create an empty list
    # cnt_rect = []
    min = 1.0
    c_o = 0
    s_o = 0 
    a_o = 0
    c_a = 0
     
    # loop through all the contours
    for i in range(0,len(contours),1):
        # for each contour, use OpenCV to calculate the area of the contour
        cnt = contours[i]
        # get rotated rect of contour and split into components
        center, size, angle = cv2.minAreaRect(cnt)

        # x,y,w,h = cv2.boundingRect(cnt)
        rect_area = size[0]*size[1]
        cnt_area = cv2.contourArea(cnt)
        if(cnt_area < 10000):
            continue
        # cnt_rect.append((rect_area-cnt_area)/float(rect_area))
        if float(rect_area) == 0:
            rect_frac = 1.1
        else:
            rect_frac = (rect_area-cnt_area)/float(rect_area)
        # print(rect_frac)
        if rect_frac < min:
            min = rect_frac
            c_o, s_o, a_o, c_a = center, size, angle, cnt_area
 
    # Sort our list of contour areas in descending order
    # list.sort(cnt_rect)
    return c_o, s_o, a_o, c_a

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

    # x,y,w,h = contour_area(contours)
    # out=image[y:y+h,x:x+w].copy()
 
    # return out

    # get rotated rect of contour and split into components
    center, size, angle, c_a = contour_area(contours)

    if c_a < 1000:
        return False, 0, 0

    # not sure why this is needed, see 
    # http://felix.abecassis.me/2011/10/opencv-rotation-deskewing/
    if angle < -45.0:
            angle += 90.0
            width, height = size[0], size[1]
            size = (height, width)
    elif angle > 45.0:
            angle -= 90.0
            width, height = size[0], size[1]
            size = (height, width)

    M = cv2.getRotationMatrix2D(center, angle, 1.0)

    # rotate the entire image around the center of the parking cell by the
    # angle of the rotated rect
    # codereview: not sure why it was necessary to swap width and height here,
    # probably related to the fact that we did angle += 90 earlier
    imgWidth, imgHeight = (image.shape[0], image.shape[1])
    rotated = cv2.warpAffine(image, M, (imgHeight, imgWidth), flags=cv2.INTER_CUBIC)

    print("Area: " + str(c_a))
    
    # extract the rect after rotation has been done
    sizeInt = (np.int0(size[0]), np.int0(size[1]))
    uprightRect = cv2.getRectSubPix(rotated, sizeInt, center)
    return True, uprightRect, c_a


def main(args):
    imgs_list, _ = get_image_list(args.image_path)

    # collect dynamic shape by auto_tune
    if use_auto_tune(args):
        tune_img_nums = 10
        auto_tune(args, imgs_list, tune_img_nums)

    # create and run predictor
    predictor = Predictor(args)
    out_int, names = predictor.run(imgs_list)

    if use_auto_tune(args) and \
        os.path.exists(args.auto_tuned_shape_file):
        os.remove(args.auto_tuned_shape_file)

    if args.benchmark:
        predictor.autolog.report()
    
    return out_int, names


if __name__ == '__main__':
    args = parse_args()
    main(args)
