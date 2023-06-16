# Agronify Model - Classification of Diseases in Plants and Ripeness Fruits Detection using CNN

This project aims to classify diseases in plants and detect the ripeness level of fruits using Convolutional Neural Network (CNN) in Python and TensorFlow. The project utilizes image data to identify plant diseases and classify the ripeness level of fruits based on the given images.

## [ 1 ] Dataset

The dataset used consists of images of plant diseases and fruits with corresponding labels. The dataset is divided into two main categories: "Classification of Diseases in Plants" and "Ripeness Fruits Detection."

Dataset Kaggle : https://kaggle.com/datasets/3610a14d2bb20a8bf03afa10859dc6642461f317c9a8eb23e99379ad91ce2b07

### **Classification of Diseases in Plants** 
This dataset contains images of plant leaves infected with diseases, and we have 2 Types of Diseased Plants, such as :

#### a. Field Crops (known as _Tanaman Pertanian_)
- :corn: Corn ( Jagung ) :leaves:- [Common Rust, Gray Leaf Spot, Healthy, Northern Leaf Blight]
- 🍁: Cassava ( Singkong ) :leaves: - [Bacterial Blight, Brown Streak Disease, Green Mottle, Healthy, Mosaic Disease]
- :tomato: Tomato ( Tomat ) :leaves: - [Bacterial Spot, Early Blight, Healthy, Late Blight, Leaf Mold, Mosaic Virus, Septoria Leaf Spot, Spider Mites, Target Spot, Yellow Leaf Curl Virus]

#### b. Plantation Crops (know as _Tanaman Perkebunan_)
- :apple: Apple ( Apel ) :leaves: - [Black Rot, Healthy, Rust, Scab]
- :sweet_potato: Potato ( Potato ) :leaves: - [Early Blight, Healthy, Late Blight]
- :grapes: Grape ( Anggur ) :leaves: - [Black Rot, ESCA, Healthy, Leaf Blight]

### **Ripeness Fruits Detection**: 
This dataset contains images of fruits at various ripeness levels (unripe and ripe), such as :
#### a. Fruits
- 🍌 Pisang - [Mentah, Matang]

#### b. Vegetables
- :tomato: Tomat - [Mentah, Matang]

## [ 2 ] Research Method
![Screenshot 2023-06-16 080644](https://github.com/Agronify/Agronify-model/assets/71364076/c8573311-a0b1-40cb-83d6-d872c30e32e7)


## [ 3 ] Model Architecture

A Convolutional Neural Network (CNN) model is used to classify both dataset categories. The model architecture can be customized based on requirements, but for this project, the following architecture is used:

- Input Layer
- Convolutional Layers: Used for feature extraction from the images.
- Max Pooling Layers: Used for dimensionality reduction of the features.
- Flatten Layer: Flattens the features into a vector.
- Fully Connected Layers: Perform classification tasks.
- Output Layer: Outputs the classification predictions, it depends on labels.

#### The Results
#### a. Disease Plants Detection   
| Model | Accuracy |  Loss |
| :---: | :---: | :---: |
| `Corn ( Jagung )` | **97%** | **9%** |
| `Apple ( Apel )` | **99%**  | **6%** |
| `Grape ( Anggur )` | **98%**  | **4%** |
| `Potato ( Kentang )` | **98%**  | **9%** |
| `Cassava ( Singkong )` | **89%**  | **27%** |
| `Tomato ( Tomat )` | **98%**  | **6%** |
| `Soybean ( Tomat )` | **98%**  | **6%** |

#### b. Ripeness Fruits Detection   
| Model | Accuracy |  Loss |
| :---: | :---: | :---: |
| `Banana ( Pisang )` | **99%**  | **9%** |
| `Tomato ( Tomat )` | **99%**  | **2%** |


  

## [ 4 ] Requirements

- Python 3 - v3.10
- Keras - v2.12.0
- TensorFlow - v2.12.0
- NumPy - v1.23.5
- Matplotlib - v3.6.3

## [ 5 ] Usage

1. Clone this repository to your local machine.
2. Install all the required dependencies.
3. Make sure the dataset is available and placed in the appropriate directory structure.
4. Run the script to train the model and perform the evaluation.
5. You can customize the script, model architecture, or hyperparameters according to your needs.

## [ 6 ] References

[1] Convolutional Neural Networks (CNNs) - Stanford University, https://cs231n.github.io/convolutional-networks/

[2] TensorFlow Documentation, https://www.tensorflow.org/api_docs

[3] Dataset

[4] Paper for Pretrained Model

## [6.] Authors

This project is developed by C23-PS050 Team Bangkit as part of Bangkit Product Capstone.
1. M305DSX2364 - Muhammad Dafa Ardiansyah 
2. M132DSX1278 - Rais Ilham Nustara  
3. M038DKY4284 - Sarah Alissa Putri

