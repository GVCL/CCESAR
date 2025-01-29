## CCESAR Dataset

This folder contains the dataset used for training and testing the models in the CCESAR project. The dataset is structured into separate folders for 8-bit and 32-bit SAR images, each containing training and testing datasets.

## Dataset Structure
After downloading the dataset, organize it as follows:

```
dataset/
│-- 8_bit_train/
│   ├── natural/
│   │   ├── images/  # Contains training images (1.tiff, 2.tiff, ...)
│   │   ├── masks/   # Corresponding ground truth masks (1_mask.tiff, 2_mask.tiff, ...)
│   ├── developed/
│       ├── images/
│       ├── masks/
│-- 8_bit_test/
│   ├── natural/
│   │   ├── images/
│   │   ├── masks/
│   ├── developed/
│       ├── images/
│       ├── masks/
│-- 32_bit_train/
│   ├── natural/
│   │   ├── images/
│   │   ├── masks/
│   ├── developed/
│       ├── images/
│       ├── masks/
│-- 32_bit_test/
│   ├── natural/
│   │   ├── images/
│   │   ├── masks/
│   ├── developed/
│       ├── images/
│       ├── masks/
```

### Naming Convention
- Each **image** file follows the format: `1.tiff`, `2.tiff`, ..., up to the total number of images in that folder.
- Each **mask** file follows the format: `1_mask.tiff`, `2_mask.tiff`, corresponding to the respective image.

## Downloading the Dataset
Due to storage constraints, we cannot provide the complete training dataset. However, you can generate your own dataset by following the instructions and downloading the required files from the link below:

[Download Training Dataset Instructions](https://drive.google.com/file/d/1jPK2MXFtgTPqbxE3blIpYcOy0WbMyOiq/view)

The training dataset consists of:
- **8-bit images**: 200 images each for `natural` and `developed`
- **32-bit images**: 200 images each for `natural` and `developed`

### Test Dataset
The test dataset is available for direct download here:

[Download Test Dataset](https://drive.google.com/drive/folders/1SR_7V9IlosRvSZrRc-bwcmArO41nHjmo)

It contains:
- **8-bit images**: 40 images each for `natural` and `developed`
- **32-bit images**: 40 images each for `natural` and `developed`

Ensure that you place the downloaded files in the appropriate folder structure as described above before running any scripts.

## Notes
- If you are using your own dataset, ensure that the images and masks follow the same structure and naming conventions.
- The dataset is essential for training and evaluating the classification and coastline extraction models used in the CCESAR project.
