# DeepfakeDetection

## Overview
DeepfakeDetection is a TensorFlow-based deepfake detection project that leverages the EfficientNet B5 architecture. The primary objective is to investigate the impact of using a dataset enhanced with super-resolution techniques, specifically Real-ESRGAN, on the performance of the detection model.

## Dataset
The project utilizes the Celeb-DF (v2) dataset, a comprehensive and widely used collection for deepfake detection tasks. This dataset contains high-quality deepfake videos featuring various celebrities, making it suitable for training and evaluating robust detection models.

## Methodology
The project workflow includes the following key steps:

1. **Data Preprocessing**
   - Original Celeb-DF (v2) videos are preprocessed and converted into frames.
   - YOLO is used to perform face cropping, ensuring that only relevant facial regions are considered.
   - Super-resolution is applied to the cropped frames using Real-ESRGAN to enhance image quality.

2. **Model Architecture**
   - EfficientNet B5 is selected for its balance between computational efficiency and performance.
   - The model is fine-tuned on the Celeb-DF (v2) dataset to optimize for the binary classification task of detecting real vs. fake videos.

3. **Training and Evaluation**
   - Two separate experiments are conducted:
     - Training and evaluation with the original dataset.
     - Training and evaluation with the super-resolved dataset.
   - Metrics such as accuracy, precision, recall, and F1-score are used to assess performance.

4. **Comparison of Results**
   - The project aims to determine whether the use of super-resolution improves the model's ability to detect deepfakes.

## Requirements
- Python 3.8 or later
- TensorFlow 2.x
- Real-ESRGAN (for super-resolution)
- YOLO (for face cropping)
- OpenCV (for frame extraction and processing)
- Additional dependencies listed in `requirements.txt`

## Installation
1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/DeepfakeDetection.git
   cd DeepfakeDetection
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up Real-ESRGAN by following the [official instructions](https://github.com/xinntao/Real-ESRGAN).

4. Set up YOLO by following the [official instructions](https://github.com/AlexeyAB/darknet).


## Acknowledgments
- The authors of the Celeb-DF (v2) dataset for providing an invaluable resource for deepfake detection.
- The developers of Real-ESRGAN for their outstanding super-resolution implementation.
- The developers of YOLO for their high-performance object detection framework.
- TensorFlow and EfficientNet teams for their state-of-the-art machine learning frameworks and models.

