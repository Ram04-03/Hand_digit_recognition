## Handwritten Digit Recognition with TensorFlow and MNIST

This project trains a Convolutional neural network model using Tensorflow to recognize handwritten digits(0-9) from the MNIST dataset using TensorFlow. 

### Project Structure

* `DigitRecognition.ipynb`: IPython Notebook file explains the CNN model which has been built
* `Trained_model_cnn.h5`: The saved trained model
* `hdr.py`: Python script for building webapp
* `requirements.txt`: List of required Python libraries
* `README.md`: This file (you're reading it now!)

### Prerequisites

* Python 3.7+
* TensorFlow 2.13
* NumPy, Matplotlib, Pillow
* Streamlit
  
### Installation

1. Clone this repository
2. Install required libraries: `pip install -r requirements.txt`
3. Load the mnist dataset using `tensorflow.keras.datasets.mnist`

### Running the project

1. Train and Test the model
2. Save the model weights
3.  Webapp for executing our model
4.  Host the webapp in streamlit
   
### Results

* Training and evaluation results will be stored
* A web application for real-time digit recognition has been built using streamlit
* The link for the webapp created https://digit-recognition.streamlit.app/



