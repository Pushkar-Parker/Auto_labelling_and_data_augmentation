# **Auto Labeling and Data Augmentation for Object Detection**

## **Overview**
This repository provides tools for **automatic data labeling** using a pre-trained YOLO model and a **data augmentation pipeline** to enhance object detection datasets. These tools help streamline dataset creation, minimize manual effort, and improve model performance.

---

## **Auto Labeling**
This module uses a trained YOLO model to **automatically label** new images by performing inference and generating bounding box annotations. This significantly **reduces manual labeling efforts** and improves efficiency.

### **Key Features:**
- Uses a trained YOLO model to **predict and generate bounding box annotations** for new images
- Supports **batch processing** for labeling multiple images at once
- Outputs annotations in a format compatible with YOLO training pipelines
- Reduces annotation time, making dataset creation **faster and more scalable**

---

## **Data Augmentation**
The data augmentation module enhances an existing dataset by applying a variety of transformations to both images and their corresponding labels, improving model generalization.

### **Key Features:**
- **Preserves annotation integrity** while applying transformations
- Supports augmentation techniques such as:
  - **Random rotation, flipping, scaling, and cropping**
  - **Color adjustments (brightness, contrast, saturation, hue shifting)**
  - **Gaussian noise and blur for robustness**
- Outputs augmented images along with their **transformed annotations**

---

## **Installation**
To run this project, install the required dependencies using the provided `requirements.txt` file:

```bash
pip install -r requirements.txt
```

---

## **Contributions**
Contributions and improvements are welcome! Feel free to fork this repository, create issues, or submit pull requests.

---

## **License**
This project is licensed under the **MIT License**.

---

### Let's Connect
If you're working on similar projects or have ideas for improvement, feel free to reach out!
