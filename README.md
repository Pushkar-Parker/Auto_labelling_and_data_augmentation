# **Auto Labeling and Data Augmentation for Object Detection**  

## **Auto Labeling**  
This repository includes a script for **automatic data labeling** using a pre-trained YOLO model. The model, trained on a custom dataset, can be leveraged to label new, unseen images by performing inference and generating annotations automatically. This process significantly **reduces manual labeling efforts**, minimizes human intervention, and improves efficiency in dataset creation for object detection tasks.  

### **Key Features:**  
- Uses a trained YOLO model to **predict and generate bounding box annotations** for new images  
- Supports **batch processing** for labeling multiple images at once  
- Outputs annotations in a format compatible with YOLO training pipelines  
- Reduces annotation time, making dataset creation **faster and more scalable**  

---

## **Data Augmentation**  
The repository also provides a **data augmentation pipeline** that enhances an existing dataset by applying a variety of transformations to both images and their corresponding labels. Augmenting the dataset helps improve the generalization capability of object detection models by increasing data diversity.  

### **Key Features:**  
- **Preserves annotation integrity** while applying transformations  
- Supports augmentation techniques such as:
  - **Random rotation, flipping, scaling, and cropping**  
  - **Color adjustments (brightness, contrast, saturation, hue shifting)**  
  - **Gaussian noise and blur for robustness**  
- Outputs augmented images along with their **transformed annotations**  

This repository is designed to streamline dataset preparation for YOLO-based object detection models, making it easier to scale labeling efforts and improve model performance.  

Would you like me to add instructions for installation, usage, and examples?
