# Agronify Model - Classification of Diseases in Plants and Ripeness Fruits Detection using CNN

This project aims to classify diseases in plants and detect the ripeness level of fruits using Convolutional Neural Network (CNN) in Python and TensorFlow. The project utilizes image data to identify plant diseases and classify the ripeness level of fruits based on the given images.

## [1] Dataset

The dataset used consists of images of plant diseases and fruits with corresponding labels. The dataset is divided into two main categories: "Classification of Diseases in Plants" and "Ripeness Fruits Detection."

### **Classification of Diseases in Plants** 
This dataset contains images of plant leaves infected with diseases, and we have 2 Types of Diseased Plants, such as :

#### a. Field Crops (known as _Tanaman Pertanian_)
- Jagung - [Common Rust, Gray Leaf Spot, Healthy, Northern Leaf Blight]
- Singkong - [Bacterial Blight, Brown Streak Disease, Green Mottle, Healthy, Mosaic Disease]
- Cabai - [Healthy, Leaf Curl, Leaf Spot, Whitefly, Yellowwish]
- Padi - [Brown Spot, Healthy, Hispa, Leaf Blast, Neck Blast]

#### b. Plantation Crops (know as _Tanaman Perkebunan_)
- Apel - [Black Rot, Healthy, Rust, Scab]
- Kentang - [Early Blight, Healthy, Late Blight]

### **Ripeness Fruits Detection**: 
This dataset contains images of fruits at various ripeness levels (unripe and ripe), such as :
#### a. Fruits
- Pisang - [Mentah, Matang]
- Apel - [Mentah, Matang]

#### b. Vegetables
- Tomat - [Mentah, Matang]

## [2.] Model Architecture

A Convolutional Neural Network (CNN) model is used to classify both dataset categories. The model architecture can be customized based on requirements, but for this project, the following architecture is used:

- Input Layer
- Convolutional Layers: Used for feature extraction from the images.
- Max Pooling Layers: Used for dimensionality reduction of the features.
- Flatten Layer: Flattens the features into a vector.
- Fully Connected Layers: Perform classification tasks.
- Output Layer: Outputs the classification predictions, it depends on labels.

#### The Results 
| Model | Accuracy |
| :---: | :---: |
| `Corn ( _Jagung_ )` | **93.4 %** |
| `Apple ( _Apel_ )` | **97.6%**  |
| `Grape ( _Anggur_ )` | **98%**  |
| `Potato ( _Kentang_ )` | **94%**  |
| `Banana ( _Pisang_ )` | **99%**  |
| `Tomato ( _Tomat_ )` | **99%**  |
  

## [3.] Requirements

- Python 3
- TensorFlow
- NumPy
- Matplotlib

## [4.] Usage

1. Clone this repository to your local machine.
2. Install all the required dependencies.
3. Make sure the dataset is available and placed in the appropriate directory structure.
4. Run the script to train the model and perform the evaluation.
5. You can customize the script, model architecture, or hyperparameters according to your needs.

## [5.] References

[1] Convolutional Neural Networks (CNNs) - Stanford University, https://cs231n.github.io/convolutional-networks/

[2] TensorFlow Documentation, https://www.tensorflow.org/api_docs

[3] Dataset

[4] Paper for Pretrained Model

## [6.] Authors

This project is developed by C23-PS050 Team Bangkit as part of Bangkit Product Capstone.
1. M305DSX2364 - Muhammad Dafa Ardiansyah 
2. M132DSX1278 - Rais Ilham Nustara  
3. M038DKY4284 - Sarah Alissa Putri

