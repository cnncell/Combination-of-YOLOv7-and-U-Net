import cv2
import numpy as np
from PIL import Image
import pandas as pd
from frcnn import FRCNN
import os  

#---------------------------------------------------#
#   template matching
#---------------------------------------------------#
def template_matching_and_save_center(img):
            template_image_path = "template/tp5.jpg"
            csv_path = 'center_coordinates.csv'
            src_image = Image.open(img)
            src_image = np.array(src_image)
            template_image = Image.open(template_image_path)
            template_image = np.array(template_image)
            g_nMatchMethod = 5  
            resultImage_rows = src_image.shape[0] - template_image.shape[0] + 1
            resultImage_cols = src_image.shape[1] - template_image.shape[1] + 1
            g_resultImage = np.zeros((resultImage_rows, resultImage_cols, 3), np.uint8)
            g_resultImage_gray = cv2.matchTemplate(src_image[:, :, 0], template_image[:, :, 0], g_nMatchMethod)
            cv2.normalize(g_resultImage_gray, g_resultImage_gray, 0, 1, cv2.NORM_MINMAX)
            minValue, maxValue, minLocation, maxLocation = cv2.minMaxLoc(g_resultImage_gray)
            matchLocation = maxLocation  
            center_x = matchLocation[0] + template_image.shape[1] // 2
            center_y = matchLocation[1] + template_image.shape[0] // 2
            with open(csv_path, 'w') as f:
                pass
            data = {'x': [center_x], 'y': [center_y]}
            df = pd.DataFrame(data)
            df.to_csv(csv_path, index=False)
            result_image = Image.fromarray(src_image)
            return result_image   
def process_images_in_folder(folder_path, output_folder=None, crop=True, count=True):  

    if not os.path.exists(folder_path):  
        print(f"Error: The folder {folder_path} does not exist.")  
        return  
  
    if output_folder and not os.path.exists(output_folder):  
        os.makedirs(output_folder)  
  
    for filename in os.listdir(folder_path):  
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):  
            img_path = os.path.join(folder_path, filename)  
            print(f"Processing {img_path}...")        
            img = Image.open(img_path)   
            detected_images = frcnn.detect_image(img, crop = crop, count = count)  
              


#---------------------------------------------------#
#  main function
#---------------------------------------------------#     
if __name__ == "__main__":
    frcnn = FRCNN()
    crop= True
    count= True                
  
    folder_path = 'jpg2'  
    output_folder = 'None'   
    process_images_in_folder(folder_path, output_folder=output_folder, crop=True, count=True) 