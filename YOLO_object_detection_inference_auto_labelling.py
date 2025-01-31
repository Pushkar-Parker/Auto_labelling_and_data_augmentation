# Importing modules

from ultralytics import YOLO
import random
import numpy as np
import os
import torch
from pathlib import Path
import cv2 as cv

random.seed(101)

# Defining colors for classes later used for object detection trackers
def classes_color(classes: dict):
    color_dict = {} # Dictionary for storing classes with their corresponding RGB values

    for i in range(len(classes)): # Looping through length of the classes
        random.seed(i) # Setting up the seed for reproducibilty of RGB values
        color_dict[i] = random.sample(range(256), 3) # Storing the RGB values
        
    return color_dict

# object detection inference and auto labeling
def object_detection(model_path: str, # Trained model path
                     test_images_path: str, # Test images
                     pred_save_path: str, # Save path for predictions
                     auto_labeling= False): # Auto labeling command

    device = 'cuda' if torch.cuda.is_available() else 'cpu' # Device agnostic code

    def get_model(model_path: str): # Getting the model
        model = YOLO(model_path).to(device)
        return model

    model = get_model(model_path) 

    test_images_path = Path(test_images_path) # Test images path
    test_images = list(test_images_path.glob('*.jpg')) # Putting '.jpg' files from the path in a list

    # Setting up labels folder and path
    labels_save_folder = "labels"
    labels_save_dir = os.path.join(pred_save_path, labels_save_folder)

    # Setting up Prediction save path
    if not os.path.exists(pred_save_path):
        os.mkdir(pred_save_path)

    # Enumerating through test images and using attributes of the predictions
    for i, image in enumerate(test_images):
        
        results = model(image) # Performing inference on an image
        
        img = results[0].orig_img # Getting the original image
        boxes = results[0].boxes.xyxy.cpu().numpy().astype(np.int32) # Getting the bounding box co-ordinates ((x0, y0), (x1, y1)) format
        labels = results[0].boxes.xywhn.cpu().numpy() # Getting the bounding box co-ordinates (x, y, w, h) format
        classes = results[0].boxes.cls.cpu().numpy().astype(np.int32) # Classes represented by integers
        classes_name = results[0].names # Classes dictionary
        labels_color = classes_color(classes_name) # Unique color specific to classes
        
        image_filename = f'{test_images[i].stem}.jpg' # Defining the image name
        txt_filename = f'{test_images[i].stem}.txt'

        # Enumerating through boxes and classes to draw rectangles and put labels on the image
        for i, (bbox, cls) in enumerate(zip(boxes, classes)):
    
            label = classes_name[cls] # Getting class name label
            label_margin = 3 # Setting up a margin

            # Getting the the text size given the parameters
            label_size = cv.getTextSize(label, # label data
                                        fontFace=cv.FONT_HERSHEY_SIMPLEX, # Font
                                        fontScale=2.5, # Font size
                                        thickness=5) # Font thickness

            # Getting the label width and hight and setting up the margins
            label_w, label_h = label_size[0]
            label_w += 2*label_margin
            label_h += 2*label_margin

            # Drawing the rectangles using predicted co-ordinates
            cv.rectangle(img, 
                         (bbox[0], bbox[1]), 
                         (bbox[2], bbox[3]), 
                         color= labels_color[cls], 
                         thickness=5)

            # Drawing the label tag to put text upon
            cv.rectangle(img, 
                         (bbox[0], bbox[1]), 
                         (bbox[0]+label_w, bbox[1]-label_h), 
                         color= labels_color[cls], 
                         thickness=-1)

            # Putting the text upon the label tag
            cv.putText(img, 
                       label, 
                       (bbox[0]+label_margin, bbox[1]-label_margin), 
                       fontFace=cv.FONT_HERSHEY_SIMPLEX, 
                       color=(255, 255, 255), 
                       fontScale= 2.5, 
                       thickness=8)

            # Defining the function of autolabeling 
            if auto_labeling is True:
                
                if not os.path.exists(labels_save_dir): # Creating the directory for saving labels
                    os.mkdir(labels_save_dir)
    
                txt_file_path = os.path.join(labels_save_dir, txt_filename)
                
                # Creating a text file and writing the data in it
                with open(txt_file_path, 'a') as f:
                    f.write(f"{cls} {labels[i][0]} {labels[i][1]} {labels[i][2]} {labels[i][3]}\n")
                f.close()
        
                print(f'Co-ordinates for {cls} is wriiten in {txt_filename} and saved in {labels_save_dir}')

        # Saving the image in the designated path
        cv.imwrite(os.path.join(pred_save_path, image_filename), img)
        print(f'{image_filename} saved in {pred_save_path}')

    # Creating the 'classes.txt'
    if auto_labeling is True:
        # Defining the 'classes.txt' file
        class_filename = 'classes.txt'
        class_file_path = os.path.join(labels_save_dir, class_filename)
    
        # writing the classes data in the txt file
        for cls in classes_name:
            with open(class_file_path, 'a') as f:
                f.write(f'{classes_name[cls]}\n')
            f.close()
        
        print(f'\n{class_filename} has been saved in {labels_save_dir}')