# Lung and Colon Cancer Histopathological Image Classification

##Project Overview
This project uses deep learning to classify histopathological images into five categories of lung and colon tissue.

The objective is to explore image classification using a custom Convolutional Neural Network (CNN) and transfer learning with MobileNetV2.

## Dataset
Dataset: Lung and Colon Cancer Histopathological Images

Source: https://www.kaggle.com/datasets/andrewmvd/lung-and-colon-cancer-histopathological-images

Total images: 25,000

Classes:
- colon_aca
- colon_n
- lung_aca
- lung_n
- lung_scc

## Models and Results

| Model | Test Accuracy |
|---|---:|
| Custom CNN | 98.51% |
| MobileNetV2 | 98.16% |

MobileNetV2 was selected for the Streamlit application.

## Technologies Used
- Python
- TensorFlow / Keras
- NumPy and Pandas
- Matplotlib and Seaborn
- Scikit-learn
- Streamlit
- Jupyter Notebook

## Project Structure
- `notebooks/`: Model development, EDA and evaluation.
- `models/`: Trained MobileNetV2 model.
- `app/`: Streamlit application.
- `results/`: Evaluation reports, visualisations and application screenshots.

## Run the Application

1. Install Python and the required packages.
2. Open a terminal in the project folder.
3. Install dependencies:

   `python -m pip install -r requirements.txt`

4. Start the application:

   `python -m streamlit run app/app.py`

5. Upload a histopathological image and click Analyse Image.

## Limitations and Disclaimer
This project is developed for experimental and academic purposes only. The predictions are not medical diagnoses and must not be used for clinical decisions. Consult qualified medical professionals for diagnosis and treatment.

The model's evaluation uses an image-level stratified split. Patient- or specimen-level independence could not be verified, so potential data leakage is a limitation.