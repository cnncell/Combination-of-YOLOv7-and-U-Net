import colorsys
import os
import time
import math 
import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageDraw, ImageFont, ImageEnhance  


from nets.yolo import YoloBody
from utils.utils import (cvtColor, get_anchors, get_classes, preprocess_input,
                         resize_image, show_config)
from utils.utils_bbox import DecodeBox, DecodeBoxNP
import csv
from unet import Unet
import pandas as pd
import random
unet = Unet()

name_classes    = ["background", "cell"]  
'''
Notes for training your own dataset!
'''
class YOLO(object):
    _defaults = {
        #--------------------------------------------------------------------------#
        #   When using your own trained model for prediction, be sure to modify model_path and classes_path!
        #   model_path points to the weight file under the logs folder, and classes_path points to the txt under model_data
        #
        #   After training, multiple weight files exist in the logs folder; select the one with lower validation set loss.
        #   Lower validation set loss does not mean higher mAP, but only that the weight has better generalization performance on the validation set.
        #   If a shape mismatch occurs, also pay attention to modifying the model_path and classes_path parameters during training
        #--------------------------------------------------------------------------#
        "model_path"        : 'logs/Y0502.pth',
        "classes_path"      : 'model_data/voc_classes.txt',
        #---------------------------------------------------------------------#
        #   anchors_path represents the txt file corresponding to the prior boxes, generally not modified.
        #   anchors_mask helps the code find the corresponding prior boxes, generally not modified.
        #---------------------------------------------------------------------#
        "anchors_path"      : 'model_data/yolo_anchors.txt',
        "anchors_mask"      : [[6, 7, 8], [3, 4, 5], [0, 1, 2]],
        #---------------------------------------------------------------------#
        #   Size of the input image, must be a multiple of 32.
        #---------------------------------------------------------------------#
        "input_shape"       : [640, 640],
        #------------------------------------------------------#
        #   Versions of YOLOv7 used in this repository, providing two options:
        #   l : corresponding to YOLOv7
        #   x : corresponding to YOLOv7_x
        #------------------------------------------------------#
        "phi"               : 'l',
        #---------------------------------------------------------------------#
        #   Only prediction boxes with scores greater than the confidence level will be retained
        #---------------------------------------------------------------------#
        "confidence"        : 0.5,
        #---------------------------------------------------------------------#
        #   IoU value for non-maximum suppression
        #---------------------------------------------------------------------#
        "nms_iou"           : 0.2,
        #---------------------------------------------------------------------#
        #   This variable controls whether to use letterbox_image for distortion-free resizing of the input image.
        #   After multiple tests, it was found that resizing directly without letterbox_image yields better results
        #---------------------------------------------------------------------#
        "letterbox_image"   : False,
        #-------------------------------#
        #   Whether to use Cuda
        #   Set to False if no GPU is available
        #-------------------------------#
        "cuda"              : True,
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    #---------------------------------------------------#
    #   Initialize YOLO
    #---------------------------------------------------#
    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        for name, value in kwargs.items():
            setattr(self, name, value)
            self._defaults[name] = value 
            
        #---------------------------------------------------#
        #   Get the number of classes and prior boxes
        #---------------------------------------------------#
        self.class_names, self.num_classes  = get_classes(self.classes_path)
        self.anchors, self.num_anchors      = get_anchors(self.anchors_path)
        self.bbox_util                      = DecodeBox(self.anchors, self.num_classes, (self.input_shape[0], self.input_shape[1]), self.anchors_mask)

        #---------------------------------------------------#
        #   Set different colors for bounding boxes
        #---------------------------------------------------#
#        hue_green = 0.33  # Hue value for green in HSV
#        hsv_tuples = [(hue_green, 1., 1.) for _ in range(self.num_classes)]
        hsv_tuples = [(x / self.num_classes, 1., 1.) for x in range(self.num_classes)]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), self.colors))
        self.generate()

        show_config(**self._defaults)

    #---------------------------------------------------#
    #   Generate the model
    #---------------------------------------------------#
    def generate(self, onnx=False):
        #---------------------------------------------------#
        #   Build the YOLO model and load the YOLO model weights
        #---------------------------------------------------#
        self.net    = YoloBody(self.anchors_mask, self.num_classes, self.phi)
        device      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.net.load_state_dict(torch.load(self.model_path, map_location=device))
        self.net    = self.net.fuse().eval()
        print('{} model, and classes loaded.'.format(self.model_path))
        if not onnx:
            if self.cuda:
                self.net = nn.DataParallel(self.net)
                self.net = self.net.cuda()

    #---------------------------------------------------#
    #   Detect images
    #---------------------------------------------------#
    def detect_image(self, image, crop=False, count=False):
        #---------------------------------------------------#
        #   Calculate the height and width of the input image
        #---------------------------------------------------#
        image_shape = np.array(np.shape(image)[0:2])
        #---------------------------------------------------------#
        #   Convert the image to an RGB image here to prevent errors during prediction with grayscale images.
        #   The code only supports prediction for RGB images; all other types of images will be converted to RGB
        #---------------------------------------------------------#
        image       = cvtColor(image)
        #---------------------------------------------------------#
        #   Add gray bars to the image for distortion-free resizing
        #   Can also directly resize for recognition
        #---------------------------------------------------------#
        image_data  = resize_image(image, (self.input_shape[1], self.input_shape[0]), self.letterbox_image)
        #---------------------------------------------------------#
        #   Add the batch_size dimension
        #   h, w, 3 => 3, h, w => 1, 3, h, w
        #---------------------------------------------------------#
        image_data  = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)), 0)

        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda()
            #---------------------------------------------------------#
            #   Input the image into the network for prediction!
            #---------------------------------------------------------#
            outputs = self.net(images)
            outputs = self.bbox_util.decode_box(outputs)
            #---------------------------------------------------------#
            #   Stack the prediction boxes and then perform non-maximum suppression
            #---------------------------------------------------------#
            results = self.bbox_util.non_max_suppression(torch.cat(outputs, 1), self.num_classes, self.input_shape, 
                        image_shape, self.letterbox_image, conf_thres=self.confidence, nms_thres=self.nms_iou)
                                                    
            if results[0] is None: 
                return image

            top_label   = np.array(results[0][:, 6], dtype='int32')
            top_conf    = results[0][:, 4] * results[0][:, 5]
            top_boxes   = results[0][:, :4]
        #---------------------------------------------------------#
        #   Set font and border thickness
        #---------------------------------------------------------#
        font        = ImageFont.truetype(font='model_data/simhei.ttf', size=np.floor(2.5e-2 * image.size[1] + 0.5).astype('int32'))
        thickness   = int(max((image.size[0] + image.size[1]) // np.mean(self.input_shape), 1))
        #---------------------------------------------------------#
        #   Counting
        #---------------------------------------------------------#
        if count:
            print("top_label:", top_label)
            classes_nums    = np.zeros([self.num_classes])
            for i in range(self.num_classes):
                num = np.sum(top_label == i)
                if num > 0:
                    print(self.class_names[i], " : ", num)
                classes_nums[i] = num
            print("classes_nums:", classes_nums)
        #---------------------------------------------------------#
        #   Whether to crop the target
        #---------------------------------------------------------#
        #---------------------------------------------------------#
        #   Whether to crop the target
        #---------------------------------------------------------#
        if crop:
            mean_x = []
            mean_y = []
            for i, c in list(enumerate(top_label)):
                predicted_class = self.class_names[int(c)]
                if predicted_class == 'Sphere_Probe':
                    continue
                top, left, bottom, right = top_boxes[i]
                top = max(0, np.floor(top).astype('int32'))
                left = max(0, np.floor(left).astype('int32'))
                bottom = min(image.size[1], np.floor(bottom).astype('int32'))
                right = min(image.size[0], np.floor(right).astype('int32'))
                

                img_origin_crop = "img_origin_crop"
                if not os.path.exists(img_origin_crop):
                    os.makedirs(img_origin_crop)
#                dir_balance_save_path = "img_balabce_crop"
#                if not os.path.exists(dir_balance_save_path):
#                    os.makedirs(dir_balance_save_path)
                dir_uimage_save_path = "dir_uimage_save_path"
                if not os.path.exists(dir_uimage_save_path):
                    os.makedirs(dir_uimage_save_path)
                    
                dir_save_path = "img_crop"
                if not os.path.exists(dir_save_path):
                    os.makedirs(dir_save_path)     
                                   
                dir_temp_1_save_path = "img_temp_1_crop"
                if not os.path.exists(dir_temp_1_save_path):
                    os.makedirs(dir_temp_1_save_path)

                dir_temp_2_save_path = "img_temp_2_crop"
                if not os.path.exists(dir_temp_2_save_path):
                    os.makedirs(dir_temp_2_save_path)
                
#                crop_gamma_save_path = "crop_gamma_save_path"
#                if not os.path.exists(crop_gamma_save_path):
#                    os.makedirs(crop_gamma_save_path)
#                dir_gray_save_path = "dir_gray_save_path"
#                if not os.path.exists(dir_gray_save_path):
#                    os.makedirs(dir_gray_save_path)
#                dir_mie_save_path = "dir_mie_save_path"
#                if not os.path.exists(dir_mie_save_path):
#                    os.makedirs(dir_mie_save_path)
#                dir_dil_save_path = "dir_dil_save_path"
#                if not os.path.exists(dir_dil_save_path):
#                    os.makedirs(dir_dil_save_path)

                    
                if predicted_class == 'cell':
                    random_integer = random.randint(1, 100000000000)
                    a = random_integer
                    crop_image_origin = image.crop([left, top, right, bottom])
#                    crop_image_origin.save(os.path.join(img_origin_crop, "crop_" +str(a)+ "_"+str(i) + ".jpg"), quality=95, subsampling=0)
                    crop_image_origin.save(os.path.join(img_origin_crop, "crop_"+ "_"+str(i) + ".jpg"), quality=95, subsampling=0)
#####################################################################################################################
#####################                      Embed U-Net network      
####################################################################################################################  
                 
                    uimage = unet.detect_image(crop_image_origin, name_classes=name_classes)
                    uimage_32 = np.float32(uimage)
                    uimage = np.uint8(uimage_32)
                    cv2.imwrite(os.path.join(dir_uimage_save_path, "crop_" + str(i) + ".png"), uimage)
                    print("save uimage crop_" + str(i) + ".png to " + dir_uimage_save_path)
                                      
#########################################################################################################################
                    crop_image = cv2.cvtColor(uimage, cv2.COLOR_BGR2GRAY)
#                    _,crop_image = cv2.threshold(crop_image, 127, 255, cv2.THRESH_BINARY)

                    _, binary_img = cv2.threshold(crop_image, 200, 255, cv2.THRESH_BINARY)  # Use INV because cells are bright
                    
                    # Find contours  
                    contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  
                    
                    # Create a PIL image to draw the result (note: convert OpenCV image from BGR to RGB)  
                    img_pil = Image.fromarray(cv2.cvtColor(binary_img, cv2.COLOR_BGR2RGB))  
                    draw = ImageDraw.Draw(img_pil)    
                    # Initialize list to store centroids of all contours  
 
                
                    # Iterate through contours and calculate centroids  
                    for i, cnt in enumerate(contours):  
                        # Calculate moments of the contour  
                        M = cv2.moments(cnt)  
                
                        # Calculate centroid  
                        if M["m00"] != 0:  
                            mean_xs = int(M["m10"] / M["m00"])  
                            mean_ys = int(M["m01"] / M["m00"])
                              
                    mean_x.append(mean_xs)       
                    mean_y.append(mean_ys)                      
                    print('mean_x', mean_x)
                    print('mean_y', mean_y)
                                       
                else:
                    center_x_ball = (right - left) / 2
                    center_y_ball = (bottom - top) / 2
                    mean_y.append(center_y_ball)
                    mean_x.append(center_x_ball)
                    continue

#                cv2.imwrite(os.path.join(dir_save_path, "crop_" + str(i) + ".png"),img_pil)#crop_image_filtered
                #crop_image.save(os.path.join(dir_save_path, "crop_" + str(i) + ".png"), quality=95, subsampling=0)
#                print("save crop_" + str(i) + ".png to " + dir_save_path)

        # ---------------------------------------------------------#
        #   Image drawing
        # ---------------------------------------------------------#
        
        #i_own = list(enumerate(top_label))
        list_class = []
        list_x = []
        list_y = []
        list_class.clear()
        list_x.clear()
        list_y.clear()

      
        for i, c in list(enumerate(top_label)):
            predicted_class = self.class_names[int(c)]
            box = top_boxes[i]
            score = top_conf[i]
             
            top, left, bottom, right = box

            top = max(0, np.floor(top).astype('int32'))
            left = max(0, np.floor(left).astype('int32'))
            bottom = min(image.size[1], np.floor(bottom).astype('int32'))
            right = min(image.size[0], np.floor(right).astype('int32'))

                # ---------------------------------------------------------#
                #   Original output center coordinates
                # ---------------------------------------------------------#
                #center_x = (right - left) / 2 + left
                #center_y = (top - bottom) / 2 + bottom

                # ---------------------------------------------------------#
                #   Recalculated center coordinates after modification: mean_x +
                # ---------------------------------------------------------#

            center_x = mean_x[i] + left  #
            center_y = mean_y[i] + top  #                     
            print('center_x', center_x)
            print('center_y', center_y)
###############################################################################################################
###############       Get relative coordinates from template matching results         ##############################################################

            df = pd.read_csv('center_coordinates.csv')
            tx = df['x'].astype(int)
            ty = df['y'].astype(int)          
            x1 = -(center_x - tx) * 0.02
            y1 = -(center_y - ty) * 0.02
            x1 = float(x1)
            y1 = float(y1)           
            x1 = round(x1, 1)
            y1 = round(y1, 1) 
            # Write to CSV file
            with open('coordinates.csv', 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                # Write data row
                writer.writerow([y1, x1])
            print("Coordinates successfully written to coordinates.csv file")
    
##############################################################################################################  
#                      Text for the bounding box                    
##############################################################################################################                
            x1 = np.around(x1, decimals=1).item()
            y1 = np.around(y1, decimals=1).item()
#            label = f'{predicted_class}:{score:.2f}  {"num:"} {i} \n {"Relative:"} {(y1,x1)}'
            label = f''
############################################################################################################## 
            label_number = '{}'.format(i)
            class_detect = '{} '.format(predicted_class)
            draw = ImageDraw.Draw(image)
            label_size = draw.textsize(label, font)
            number_size = draw.textsize(label_number, font)

            label = label.encode('utf-8')

            if top - label_size[1] >= 0:
                    text_origin = np.array([left, top - label_size[1]])
            else:
                    text_origin = np.array([left, top + 1])

            if top - number_size[0] >= 0:
                    number_s = np.array([left, top - number_size[0]-16])#4
            else:
                    number_s = np.array([left, top + 1])

            for i in range(thickness):
                    draw.rectangle([left + i, top + i, right - i, bottom - i], outline=self.colors[c])


 
            class_data = 'Cell category:{}'.format(class_detect)
            list_class.append(class_data)

############################################################################################################
#####################   Crosshair at cell center   ######################################################################    
            # Bright yellow-green (RGB format)
            BRIGHT_GREEN = (0, 255, 0)# Vibrant apple green
            for i in range(thickness):
                draw.rectangle([center_x + 4, center_y + 20, center_x - 4, center_y - 20], 
                            fill=BRIGHT_GREEN, outline=BRIGHT_GREEN)
                draw.rectangle([center_x + 20, center_y + 4, center_x - 20, center_y - 4], 
                            fill=BRIGHT_GREEN, outline=BRIGHT_GREEN)
                draw.rectangle([left + i, top + i, right - i, bottom - i], outline=self.colors[c])      
                draw.rectangle([left + i+1, top + i+1, right - i+1, bottom - i+1], outline=self.colors[c])  

##############################################################################################################
##############          Draw template matching results      #############################################################                            
            yellow =  (255, 255, 0)
            BLUE = (0, 0, 255)                     
            draw.polygon([
                (tx, ty - 18),  # Top vertex
                (tx + 18, ty),  # Right vertex
                (tx, ty + 18),  # Bottom vertex
                (tx - 18, ty)   # Left vertex
            ], fill=yellow, outline=yellow)
            draw.rectangle([(tx + 247), (ty + 247), (tx - 247), (ty - 247)], outline=BLUE)
            draw.rectangle([(tx + 248), (ty + 248), (tx - 248), (ty - 248)], outline=BLUE)
            draw.rectangle([(tx + 249), (ty + 249), (tx - 249), (ty - 249)], outline=BLUE)
            draw.rectangle([(tx + 250), (ty + 250), (tx - 250), (ty - 250)], outline=BLUE)



      
###################################################################################################################                

            print("      ")
            print("      ")
                
            draw.rectangle([tuple(text_origin), tuple(text_origin + label_size)], fill=self.colors[c])
            draw.text(text_origin, str(label,'UTF-8'), fill=(0, 0, 0), font=font)
            del draw

        return image, top_label, count
   