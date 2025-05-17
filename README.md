# Face Mask Detection Project

This repository contains a face mask detection system developed using TensorFlow and transfer learning with MobileNetV2. The system classifies images into `with_mask` and `without_mask` categories, achieving a test accuracy of 96.62%. It includes a detailed project report and the model code for training and inference.

## Project Overview
- **Dataset**: Face Mask Dataset by Omkar Gurav, sourced from Kaggle ([link](https://www.kaggle.com/datasets/omkargurav/face-mask-dataset)).
- **Model**: MobileNetV2 with a custom head (GlobalAveragePooling2D → Dropout → Dense(2, softmax)).
- **Accuracy**: 96.62% on the test set (1511 images).

## Repository Structure
- **report/**: Contains the project report and related files.
  - `Face Mask Detection Report.pdf`: Compiled project report with figures.
  - `Face Mask Detection Report.tex`: LaTeX source for the report.
  - Figures: `dataset_distribution.png`, `model_architecture.png`, `training_progress.png`, `sample_predictions.png`.
- **model/**: Contains the model code and related files.
  - `Face Mask Detection Using TensorFlow.ipynb`: Jupyter Notebook for training the model.
  - `mask_detection_model.keras`: Trained model weights.
  - `requirements.txt`: List of Python dependencies.
- **dataset/**: Contains the image data files used to train and test the model
  -**test/**: Contains image data for testing
   -**with_mask/**: Contains with mask images for testing purpose
   -**without_mask/**: Contains without mask images for testing purpose
  -**train/**: Contains image data for training
   -**with_mask/**: Contains with mask images for training purpose
   -**without_mask/**: Contains without mask images for training purpose
  
## How to Compile the Report
1. Install MiKTeX and ensure `pdflatex` is in your PATH.
2. Navigate to the `report` directory and ensure all images are present.
3. Run: pdflatex "Face Mask Detection Report.tex"
4. Open `Face Mask Detection Report.pdf` to view the report.

## How to Run the Model
### Prerequisites
- Python 3.8 or higher.
- Jupyter Notebook (for running `train_model.ipynb`).
- Access to the dataset from Kaggle ([link](https://www.kaggle.com/datasets/omkargurav/face-mask-dataset)).

### 1. Set Up the Environment
1. Clone this repository: git clone https://github.com/harshikala/MaskDetectionProject.git
cd MaskDetectionProject
2. Navigate to the `model` directory: cd model
3. Install the dependencies: pip install -r requirements.txt

### 2. Download the Dataset
- Download the Face Mask Dataset from Kaggle ([link](https://www.kaggle.com/datasets/omkargurav/face-mask-dataset)).
- Extract the dataset and place the `with_mask` and `without_mask` folders in a directory named `dataset/` inside the `model/` directory:
  model/dataset/
├── with_mask/
└── without_mask/

- Alternatively, if you only want to test predictions, use the provided `sample_images/` directory (if available).

### 3. Train the Model (Optional)
If you want to retrain the model:
1. Ensure the dataset is placed in `model/dataset/`.
2. Open the Jupyter Notebook: Face Mask Detection Using TensorFlow.ipynb
3. Run all cells in the notebook to train the model.
4. The trained model will be saved as `mask_detection_model.keras`.

### 4. Make Predictions
To make predictions on new images using the pre-trained model:
1. Ensure `mask_detection_model.keras` is in the `model/` directory.
2. Run the prediction script: python predict_mask.py path/to/image.jpg
- Replace `path/to/image.jpg` with the path to your image.
- Example using a sample image (if `sample_images/` is present): python predict_mask.py sample_images/with_mask_820.jpg
3. The script will output the prediction (`MASK` or `NO MASK`) along with the confidence score.

## Acknowledgments
- **Dataset**: Omkar Gurav (Kaggle).
- **Model**: MobileNetV2, pre-trained on ImageNet.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
