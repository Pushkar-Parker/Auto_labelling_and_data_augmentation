import random
import os
import cv2 as cv
import albumentations as A
from pathlib import Path

# Defining a transformer
transform = A.Compose([A.Rotate(limit=(-40, 60), p=1.0),],
                      bbox_params= A.BboxParams(format= 'yolo', min_visibility=0.7))

# Getting the data after analysing it
def get_data(images_path: str, 
             labels_path: str): # Passing the images and labels folder's path

    def access_files(images_path: str, 
                     labels_path: str): # Defining function for accessing the files

        images_path = Path(images_path) # Passing the path of the images
        images = list(images_path.glob('*.jpg')) # Putting the images in a list
    
        labels_path = Path(labels_path) # Passing the path of the labels
        labels = list(labels_path.glob('*.txt')) # Putting the labels in a list

        return images, labels

    images, labels = access_files(images_path, labels_path) # Creating the images and labels instances

    # Getting names of the files for further analysis
    labels_name = [label.stem for label in labels]
    images_name = [image.stem for image in images]

    # Checking if the 'classes.txt' exist among the labels
    if 'classes' in labels_name:
        print('classes.txt exist in labels folder\nMoving further...\n')
        labels_name.remove('classes') # removing 'classes.txt' from the list making the number of images and labels the same

        # If conditions are met the data files are returned
        if len(labels_name) == len(images_name):
            print('Data is in correct format')

        # If there is ambiguity in the data it will be pointed out
        elif len(labels_name) != len(images_name):
    
            unmatched_labels = [image for image in images_name if image not in labels_name] # Checking what images do not have the corresponding '.txt' label file
            unmatched_labels_for = [label+'.jpg' for label in unmatched_labels] # Adding the image filename with the '.jpg' extension for better understanding while printing
            no_labels = ', '.join(unmatched_labels_for) # Doing split operation for better readabilty purpose
            
            unmatched_images = [label for label in labels_name if label not in images_name] # Checking which labels are extra and do not have a corresponding image
            unmatched_images_for = [image+'.txt' for image in unmatched_images] # Adding the filename with the '.txt' extension
            no_images = ', '.join(unmatched_images_for) # Doing split operation
     
            if len(unmatched_images) != 0:
                print(f"Extra labels for non existing corresponding image: {no_images}")

            if len(unmatched_labels) != 0:
                print(f"No labels found for corresponding images: {no_labels}")    
                images, labels = access_files(images_path, labels_path)

    # In case the 'classes.txt' does not exist it'll be pointed out the code will stop executing
    elif 'classes' not in labels_name:
        print(f'There is no classes.txt in {labels_path}')
        exit()
        

    labels.pop(len(labels)-1) # Removing 'classes.txt' from the labels' list to avoid any printing of 'classes.txt'
    
    return images, labels # returning the data


def augment_data(images: list, # Passing the images from get_data() 
                 labels: list, # Passing the labels from gata_data()
                 num: int, # Passing the number of files to be generated during augmentation
                 save_aug_data: str): # Passing the save path

    aug_images_folder, aug_labels_folder = 'images', 'labels' # Defining the images and labels folder

    # Creating directory for saving augmented images
    save_aug_images = os.path.join(save_aug_data, aug_images_folder) 
    if not os.path.exists(save_aug_images):
        os.mkdir(save_aug_images)

    # Creating directory for saving augmented labels
    save_aug_labels = os.path.join(save_aug_data, aug_labels_folder)
    if not os.path.exists(save_aug_labels):
        os.mkdir(save_aug_labels)
    

    index_range = range(len(images)) # Getting range from the images which will be used to randomly select data for data augmentation
    
    for i in range(num): 
        random_idx = random.choice(index_range) # Getting an index randomly from the index

        img = images[random_idx] # Getting the image using random index
        img = cv.imread(img) # Reading the image
        img_name = f'{i}.jpg' # Setting the image name

        label = labels[random_idx] # Getting the label corresponding to the image
        label_name = f'{i}.txt'# Setting the label name

        # Opening the label file and reading the coordinates
        f = open(label, 'r')
        cordinates = f.readlines()
        f.close()

        # Setting up the bounding box for drawing rectangles on the image
        bboxes = [] 
        for cor in cordinates:

            # Split string to float
            l, x, y, w, h = map(float, cor.split(' '))
            bboxes.append([x, y, w, h, int(l)])

        transformed = transform(image= img, bboxes=bboxes) # Transformed the image and labels
        transformed_cordinates = transformed['bboxes'] 
        transformed_img = transformed['image']
        aug_image = transformed_img.copy()

        dh, dw, _ = transformed_img.shape # Getting the height and width of the transformed image (HxWxC)

        for i in range(len(transformed_cordinates)): # Looping through the transformed coordinates
            
            datapoints = [] # Creating an empty list to fill with appropriate coordinates
            for j in range(len(transformed_cordinates[i])):
                datapoints.append(transformed_cordinates[i][j])
        
            l, x, y, w, h = int(datapoints[4]), datapoints[0], datapoints[1], datapoints[2], datapoints[3] # getting the coordinate values for conversion
            
            x0 = int((x - w / 2) * dw)
            x1 = int((x + w / 2) * dw)
            y0 = int((y - h / 2) * dh)
            y1 = int((y + h / 2) * dh)
            
            if x0 < 0:
                x0 = 0
            if x1 > dw - 1:
                x1 = dw - 1
            if y0 < 0:
                y0 = 0
            if y1 > dh - 1:
                y1 = dh - 1
        
            cv.rectangle(transformed_img, (x0, y0), (x1, y1), (10, 255, 10), 10) # Drawing the rectangles on the image
        
            # Writing the xywh cordinates in the label file
            label_file_path = os.path.join(save_aug_labels, label_name)
            label_file = open(label_file_path, 'a')

            # print(f'{l} {x} {y} {w} {h}')
            
            label_file.write(f'{l} {x} {y} {w} {h}\n')
            label_file.close()

        # Saving the image file
        image_aug_file = os.path.join(save_aug_images, img_name)
        image_pred_file = os.path.join(save_aug_data, img_name)
        
        cv.imwrite(image_pred_file, transformed_img)
        cv.imwrite(image_aug_file, aug_image)

        print(f'Saved transformed labeled image {img_name} in {save_aug_data}')
        print(f'Saved transformed label {label_name} in {save_aug_labels}')
        print(f'Saved transformed augmented image {img_name} in {save_aug_images}')

    # The number of files that have been saved
    print(f'{num} images and labels have been saved')

# Getting the data
images, labels = get_data(images_path= 'path', labels_path= 'path')
augment_data(images=images, labels=labels, num=8, save_aug_data='path')