import colorsys
import copy
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from torch import nn
from PIL import Image, ImageDraw  
from nets.unet import Unet as unet
from utils.utils_unet import cvtColor, preprocess_input, resize_image, show_config


#--------------------------------------------#
#   Modify two parameters when using your trained model for prediction
#   Both model_path and num_classes need to be modified!
#   If a shape mismatch occurs,
#   ensure that model_path and num_classes are adjusted during training
#--------------------------------------------#
class Unet(object):
    _defaults = {
        #-------------------------------------------------------------------#
        #   model_path points to the weight file in the logs folder.
        #   After training, multiple weight files exist in the logs folder; select the one with lower validation loss.
        #   Lower validation loss doesn't necessarily mean higher mIoU, but indicates better generalization on the validation set.
        #-------------------------------------------------------------------#
        "model_path"    : 'logs/U_922.pth',
        #--------------------------------#
        #   Number of classes to distinguish + 1
        #--------------------------------#
        "num_classes"   : 2,
        #--------------------------------#
        #   Backbone network used: vgg, resnet50   
        #--------------------------------#
        "backbone"      : "vgg",
        #--------------------------------#
        #   Size of the input image
        #--------------------------------#
        "input_shape"   : [64, 64],
        #-------------------------------------------------#
        #   mix_type controls the visualization method of detection results
        #
        #   mix_type = 0: Blend original image with generated image
        #   mix_type = 1: Only retain the generated image
        #   mix_type = 2: Remove background, retain only targets in the original image
        #-------------------------------------------------#
        "mix_type"      : 1,
        #--------------------------------#
        #   Whether to use Cuda
        #   Set to False if no GPU is available
        #--------------------------------#
        "cuda"          : True,
    }

    #---------------------------------------------------#
    #   Initialize UNET
    #---------------------------------------------------#
    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        for name, value in kwargs.items():
            setattr(self, name, value)
        #---------------------------------------------------#
        #   Set different colors for drawing
        #---------------------------------------------------#
        if self.num_classes <= 21:
            self.colors = [ (0, 0, 0), (255, 255, 255), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128), (0, 128, 128), 
                            (128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 128, 0), (192, 128, 0), (64, 0, 128), (192, 0, 128), 
                            (64, 128, 128), (192, 128, 128), (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0), (0, 64, 128), 
                            (128, 64, 12)]
        else:
            hsv_tuples = [(x / self.num_classes, 1., 1.) for x in range(self.num_classes)]
            self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
            self.colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), self.colors))
        #---------------------------------------------------#
        #   Get the model
        #---------------------------------------------------#
        self.generate()
        
        show_config(**self._defaults)

    #---------------------------------------------------#
    #   Load model weights
    #---------------------------------------------------#
    def generate(self, onnx=False):
        self.net = unet(num_classes=self.num_classes, backbone=self.backbone)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.net.load_state_dict(torch.load(self.model_path, map_location=device))
        self.net = self.net.eval()
        print(f'{self.model_path} model and classes loaded.')
        if not onnx:
            if self.cuda:
                self.net = nn.DataParallel(self.net)
                self.net = self.net.cuda()

    #---------------------------------------------------#
    #   Detect images
    #---------------------------------------------------#
    def detect_image(self, image, count=False, name_classes=None):
        #---------------------------------------------------------#
        #   Convert the image to RGB here to prevent errors with grayscale images during prediction.
        #   The code only supports RGB image prediction; all other types will be converted to RGB.
        #---------------------------------------------------------#
        image = cvtColor(image)
        #---------------------------------------------------#
        #   Create a backup of the input image for later drawing
        #---------------------------------------------------#
        old_img = copy.deepcopy(image)
        orininal_h = np.array(image).shape[0]
        orininal_w = np.array(image).shape[1]
        #---------------------------------------------------------#
        #   Add gray bars for distortion-free resizing
        #   Can also resize directly for recognition
        #---------------------------------------------------------#
        image_data, nw, nh = resize_image(image, (self.input_shape[1], self.input_shape[0]))
        #---------------------------------------------------------#
        #   Add batch_size dimension
        #---------------------------------------------------------#
        image_data = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, np.float32)), (2, 0, 1)), 0)

        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda()
                
            #---------------------------------------------------#
            #   Input image into the network for prediction
            #---------------------------------------------------#
            pr = self.net(images)[0]
            #---------------------------------------------------#
            #   Get the class of each pixel
            #---------------------------------------------------#
            pr = F.softmax(pr.permute(1, 2, 0), dim=-1).cpu().numpy()
            #--------------------------------------#
            #   Crop out the gray bar part
            #--------------------------------------#
            pr = pr[int((self.input_shape[0] - nh) // 2): int((self.input_shape[0] - nh) // 2 + nh), 
                    int((self.input_shape[1] - nw) // 2): int((self.input_shape[1] - nw) // 2 + nw)]
            #---------------------------------------------------#
            #   Resize the image to original dimensions
            #---------------------------------------------------#
            pr = cv2.resize(pr, (orininal_w, orininal_h), interpolation=cv2.INTER_LINEAR)
            #---------------------------------------------------#
            #   Get the class of each pixel
            #---------------------------------------------------#
            pr = pr.argmax(axis=-1)
        
        #---------------------------------------------------------#
        #   Count class statistics
        #---------------------------------------------------------#
        if count:
            classes_nums = np.zeros([self.num_classes])
            total_points_num = orininal_h * orininal_w
            print('-' * 63)
            print("|%25s | %15s | %15s|"%("Key", "Value", "Ratio"))
            print('-' * 63)
            for i in range(self.num_classes):
                num = np.sum(pr == i)
                ratio = num / total_points_num * 100
                if num > 0:
                    print("|%25s | %15s | %14.2f%%|"%(str(name_classes[i]), str(num), ratio))
                    print('-' * 63)
                classes_nums[i] = num
            print("classes_nums:", classes_nums)       

        if self.mix_type == 0:
            # seg_img = np.zeros((np.shape(pr)[0], np.shape(pr)[1], 3))
            # for c in range(self.num_classes):
            #     seg_img[:, :, 0] += ((pr[:, :] == c ) * self.colors[c][0]).astype('uint8')
            #     seg_img[:, :, 1] += ((pr[:, :] == c ) * self.colors[c][1]).astype('uint8')
            #     seg_img[:, :, 2] += ((pr[:, :] == c ) * self.colors[c][2]).astype('uint8')
            seg_img = np.reshape(np.array(self.colors, np.uint8)[np.reshape(pr, [-1])], [orininal_h, orininal_w, -1])
            #------------------------------------------------#
            #   Convert the new image to PIL Image format
            #------------------------------------------------#
            image = Image.fromarray(np.uint8(seg_img))
            uimage_32 = np.float32(image)
            uimage = np.uint8(uimage_32)
            crop_image = cv2.cvtColor(uimage, cv2.COLOR_BGR2GRAY)
            _, binary_img = cv2.threshold(crop_image, 127, 255, cv2.THRESH_BINARY)  # Use INV since cells are bright
              
            # Find contours  
            contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  
            
            # Create a PIL image for drawing results (convert OpenCV BGR to RGB)  
            img_pil = Image.fromarray(cv2.cvtColor(binary_img, cv2.COLOR_BGR2RGB))  
            draw = ImageDraw.Draw(old_img)    
                      
            # Iterate through contours and calculate centroids  
            for cnt in contours:  
                # Calculate contour moments  
                M = cv2.moments(cnt)  
                
                # Calculate centroid  
                if M["m00"] != 0:  
                    cX = int(M["m10"] / M["m00"])  
                    cY = int(M["m01"] / M["m00"])  
                    
                    # Draw centroid on PIL image (PIL uses top-left as origin)
                    color = (255, 0, 0)  # Red   
                    draw.ellipse([cX-2, cY-2, cX+2, cY+2], outline='red')  # Use outline instead of fill  
                    draw.rectangle([cX + 3, cY + 15, cX - 3, cY - 15], fill=color, outline=color)
                    draw.rectangle([cX + 15, cY + 3, cX - 15, cY - 3],
                                   fill=color, outline=color)            
            # Display the image  
#            img_pil.show()  
             
            image = Image.blend(old_img, img_pil, 0.2)
        elif self.mix_type == 1:
            # seg_img = np.zeros((np.shape(pr)[0], np.shape(pr)[1], 3))
            # for c in range(self.num_classes):
            #     seg_img[:, :, 0] += ((pr[:, :] == c ) * self.colors[c][0]).astype('uint8')
            #     seg_img[:, :, 1] += ((pr[:, :] == c ) * self.colors[c][1]).astype('uint8')
            #     seg_img[:, :, 2] += ((pr[:, :] == c ) * self.colors[c][2]).astype('uint8')
            seg_img = np.reshape(np.array(self.colors, np.uint8)[np.reshape(pr, [-1])], [orininal_h, orininal_w, -1])
            #------------------------------------------------#
            #   Convert the new image to PIL Image format
            #------------------------------------------------#
            image = Image.fromarray(np.uint8(seg_img))

        elif self.mix_type == 2:
            seg_img = (np.expand_dims(pr != 0, -1) * np.array(old_img, np.float32)).astype('uint8')
            #------------------------------------------------#
            #   Convert the new image to PIL Image format
            #------------------------------------------------#
            image = Image.fromarray(np.uint8(seg_img))
            
        elif self.mix_type == 3:
            seg_img = (np.expand_dims(pr != 0, -1) * np.array(old_img, np.float32)).astype('uint8')
            #------------------------------------------------#
            #   Convert the new image to PIL Image format
            #------------------------------------------------#
            image = Image.fromarray(np.uint8(seg_img))
            
            image = Image.blend(old_img, image, 0.3)
            
        if self.mix_type == 4:
            # seg_img = np.zeros((np.shape(pr)[0], np.shape(pr)[1], 3))
            # for c in range(self.num_classes):
            #     seg_img[:, :, 0] += ((pr[:, :] == c ) * self.colors[c][0]).astype('uint8')
            #     seg_img[:, :, 1] += ((pr[:, :] == c ) * self.colors[c][1]).astype('uint8')
            #     seg_img[:, :, 2] += ((pr[:, :] == c ) * self.colors[c][2]).astype('uint8')
            seg_img = np.reshape(np.array(self.colors, np.uint8)[np.reshape(pr, [-1])], [orininal_h, orininal_w, -1])
            #------------------------------------------------#
            #   Convert the new image to PIL Image format
            #------------------------------------------------#
            image = Image.fromarray(np.uint8(seg_img))
            uimage_32 = np.float32(image)
            uimage = np.uint8(uimage_32)
            crop_image = cv2.cvtColor(uimage, cv2.COLOR_BGR2GRAY)
            _, binary_img = cv2.threshold(crop_image, 127, 255, cv2.THRESH_BINARY)  # Use INV since cells are bright
              
            # Find contours  
            contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  
            
            # Create a PIL image for drawing results (convert OpenCV BGR to RGB)  
            img_pil = Image.fromarray(cv2.cvtColor(binary_img, cv2.COLOR_BGR2RGB))  
            draw = ImageDraw.Draw(img_pil)    
                      
            # Iterate through contours and calculate centroids  
            for cnt in contours:  
                # Calculate contour moments  
                M = cv2.moments(cnt)  
                
                # Calculate centroid  
                if M["m00"] != 0:  
                    cX = int(M["m10"] / M["m00"])  
                    cY = int(M["m01"] / M["m00"])  
                    
                    # Draw centroid on PIL image (PIL uses top-left as origin)
                    color = (255, 0, 0)  # Red   
                    draw.ellipse([cX-2, cY-2, cX+2, cY+2], outline='red')  # Use outline instead of fill  
                    draw.rectangle([cX + 3, cY + 15, cX - 3, cY - 15], fill=color, outline=color)
                    draw.rectangle([cX + 15, cY + 3, cX - 15, cY - 3],
                                   fill=color, outline=color)            
#            image   = Image.blend(old_img, image, 0.2)
             
#            image   = Image.blend(old_img, img_pil, 0.2)
            
        return image

