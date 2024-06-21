
# Detection of Facial Forgeries Using MesoNet

## Overview
This project focuses on detecting facial forgeries, commonly known as deepfakes, using a condensed version of the MesoNet neural network architecture. The aim is to leverage machine learning and deep learning techniques to identify altered video frames effectively.

## Table of Contents
- [Overview](#overview)
- [Project Purpose](#project-purpose)
- [Dataset](#dataset)
- [Neural Network Architecture](#neural-network-architecture)
- [Implementation](#implementation)
- [Results](#results)
- [Conclusion](#conclusion)
- [Future Work](#future-work)
- [Installation](#installation)
- [Usage](#usage)
- [How to Implement the Code](#how-to-implement-the-code)
- [Contributing](#contributing)
- [License](#license)

## Project Purpose
The primary goal of this project is to develop a robust system for detecting facial forgeries using the MesoNet architecture. The objectives include:
- Implementing machine learning and deep learning techniques.
- Evaluating the performance of various models on a specialized dataset.
- Providing insights into the effectiveness of MesoNet in real-world scenarios.

## Dataset
The dataset used in this project includes both altered (deepfake) and unaltered (authentic) video frames. We utilized the following datasets:
- [FaceForensics++](https://github.com/ondyari/FaceForensics)
- [DeepFake Detection Challenge Dataset](https://www.kaggle.com/c/deepfake-detection-challenge)

### Data Preprocessing
- Frames are extracted from videos.
- Faces are detected and aligned.
- Images are resized to 224x224 pixels.

## Neural Network Architecture
### MesoNet Architecture
MesoNet is a convolutional neural network (CNN) designed specifically for detecting facial forgeries. The condensed version of MesoNet implemented in this project includes the following layers:
- **Conv2D Layer**: Convolutional layer with ReLU activation.
- **Batch Normalization Layer**: Normalizes the output of the previous layer.
- **MaxPooling Layer**: Reduces the spatial dimensions.
- **Fully Connected Layer**: Outputs the classification results.

## Implementation
### Code Structure
- `data/`: Contains the dataset.
- `weights/`: Pre-trained model weights.
- `master_notebook.ipynb`: Jupyter Notebook for training and evaluating the model.
- `utils.py`: Utility functions for data preprocessing and augmentation.

### Training
- **Parameters**: Learning rate, epochs, batch size.
- **Hardware**: Training performed on GPU for faster computation.
- **Training Time**: Approximately 2 hours for 10 epochs on a dataset of 10,000 images.

### Evaluation
- **Metrics**: Accuracy, precision, recall, F1-score.
- **Visualization**: Confusion matrix and ROC curve.

## Results
The model achieved the following performance metrics on the test dataset:
- **Accuracy**: 92%
- **Precision**: 90%
- **Recall**: 89%
- **F1-Score**: 89%

### Key Findings
- The condensed MesoNet architecture is effective in detecting facial forgeries.
- The model performs well on both high-quality and low-quality deepfakes.

## Conclusion
This project demonstrates the capability of the MesoNet architecture in detecting deepfakes with high accuracy. The model can be further optimized for real-time applications.

## Future Work
- **Data Augmentation**: Implementing advanced data augmentation techniques to improve model robustness.
- **Model Optimization**: Exploring more efficient neural network architectures for better performance.
- **Real-Time Detection**: Adapting the model for real-time deepfake detection in video streams.

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/Saad-data/Detection-of-Facial-Forgeries-Using-MesoNet.git
   ```
2. Navigate to the project directory:
   ```bash
   cd Detection-of-Facial-Forgeries-Using-MesoNet
   ```
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
1. Run the Jupyter Notebook to train and evaluate the model:
   ```bash
   jupyter notebook master_notebook.ipynb
   ```
2. Follow the steps in the notebook to preprocess data, train the model, and evaluate its performance.

## How to Implement the Code
1. **Data Preparation**:
   - Ensure you have the required datasets downloaded and placed in the `data/` directory.
   - Run data preprocessing scripts provided in the notebook to prepare the data for training.

2. **Training the Model**:
   - Open `master_notebook.ipynb` in Jupyter Notebook.
   - Execute the cells step-by-step to preprocess the data, define the model architecture, and start the training process.
   - Adjust the training parameters such as learning rate, epochs, and batch size as needed.

3. **Evaluating the Model**:
   - After training, the notebook will guide you through the evaluation process.
   - Use the provided evaluation scripts to calculate performance metrics like accuracy, precision, recall, and F1-score.
   - Visualize the results using confusion matrices and ROC curves.

4. **Inference**:
   - Use the trained model to make predictions on new, unseen data.
   - Implement inference scripts to process new video frames and detect forgeries in real-time.

## Contributing
We welcome contributions from the community. Please follow these steps to contribute:
1. Fork the repository.
2. Create a new branch:
   ```bash
   git checkout -b feature-branch
   ```
3. Commit your changes:
   ```bash
   git commit -m "Add new feature"
   ```
4. Push to the branch:
   ```bash
   git push origin feature-branch
   ```
5. Open a pull request.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
