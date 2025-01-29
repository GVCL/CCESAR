## CCESAR: Coastline Classification-Extraction From SAR Images Using CNN-U-Net Combination

This repository contains the official implementation of the paper [*CCESAR: Coastline Classification-Extraction From SAR Images Using CNN-U-Net Combination*](https://arxiv.org/pdf/2501.12384). The repository includes code for training and testing classification models, coastline extraction, and generating ground truth masks from SAR images.

## Repository Structure

### **1. 8bit**
- Contains code for training and testing models for **8-bit SAR images**.
- Files:
  - `classification_model.py`: Defines the classification model architecture.
  - `constants.py`: Contains a list of constants used across the scripts.
  - `iou.py`: Function to calculate the IoU (Intersection over Union), given an image and mask path.
  - `process_input.py`: Helper functions used during training.
  - `train_classification.py`: Training script for the classification model.
  - `train_detection.py`: Training script for the detection model.
  - `unet_model.py`: Defines the U-Net model architecture used in the detection model.
  - `test.py`: Complete pipeline for testing the models, generating predictions, and calculating IoU.

### **2. 32bit**
- Contains code for training and testing models for **32-bit SAR images**.
- Files:
  - `classification_model.py`: Defines the classification model architecture.
  - `constants.py`: Contains a list of constants used across the scripts.
  - `iou.py`: Function to calculate the IoU (Intersection over Union), given an image and mask path.
  - `process_input.py`: Helper functions used during training.
  - `train_classification.py`: Training script for the classification model.
  - `train_detection.py`: Training script for the detection model.
  - `unet_model.py`: Defines the U-Net model architecture used in the detection model.
  - `test.py`: Complete pipeline for testing the models, generating predictions, and calculating IoU.

### **3. saved_models**
- Contains six pre-trained models (in `.h5` format) used in the paper:
  - **Classification models** for both 8-bit and 32-bit SAR images.
  - **Coastline detection models** for both natural and developed coastlines (for both 8-bit and 32-bit SAR images).
- These models can be directly used for testing without additional training.

### **4. mask_generation**
- Contains scripts to generate ground truth masks:
  - `optimisedClipVectorByExtent.py`: Clips vectors by the extent of SAR images.
  - `createTruthMask.py`: Generates ground truth masks from clipped vector data.
  - **Required file**: `land_polygon.shp` is required to run `createTruthMask.py`. It can be downloaded from [this link](https://drive.google.com/file/d/1d1skepWM0uCL96x6J4sL_KcH61s7t_VP/view).
  - **Setup**: Fill in the constants in `constants.py` inside the `mask_generation` folder before running. These include:
    - Path to the `land_polygon.shp` file
    - Path to the input images directory
    - Path to the output intermediate directory
    - Path to the output directory

### **5. dataset**
- This folder contains the dataset required for training and testing the models.
- More details on dataset setup and structure are provided in the README inside the `dataset` folder.

## Software Requirements
The code was developed and tested with the following dependencies:
- **Python**: 3.10.13
- **TensorFlow**: 2.10.0
- **Scikit-Image**: 0.25.0

## How to Use

### 1. Setting Up the Dataset
Before running the scripts, set up the `dataset` folder as instructed in the README file located in the dataset folder.

### 2. Training the Models
1. **Train the Detection Model**:
    ```bash
    python train_detection.py
    ```
2. **Train the Classification Model**:
   ```bash
   python train_classification.py
   ```
3. The trained models will be saved in the `saved_models` folder. Pre-trained models used in the paper are already provided if you want to skip training.

### 3. Testing the Models
To test the models:
```bash
python test.py
```
- The predicted images will be stored in the `results` folder.
- The IoU for each test image and the average IoU across all test images will be printed in the terminal.

### Note:
The above workflow applies to both `8_bit` and `32_bit` folders. Simply navigate to the appropriate folder and run the corresponding scripts.

