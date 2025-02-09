Neural Network for Time-Series Prediction
This repository contains the code and analysis for building, training, and evaluating neural network models to predict stock prices using historical data. The dataset used in this exercise is NVIDIA stock price data.

Overview
This project demonstrates how to preprocess time-series data, engineer features, and build a Recurrent Neural Network (RNN) to predict stock prices based on historical closing prices. The neural network is built using PyTorch. We experiment with different hyperparameters, evaluate the performance of the model, and provide insights into the results.

Project Structure
data/: Contains the raw and processed dataset.
models/: Contains the PyTorch model definitions and training scripts.
notebooks/: Jupyter notebooks for data exploration and analysis (if applicable).
README.md: This file, which provides an overview of the project.
requirements.txt: List of required Python packages.
Dataset
The dataset used for this project is NVDA stock price data, obtained from Yahoo Finance. The data contains the historical stock prices of NVDA, including:

Date: The date of the stock price data.
Open: The opening price.
High: The highest price.
Low: The lowest price.
Close: The closing price.
Volume: The trading volume.
The dataset spans the last 5 years.

Feature Engineering
Before training the model, the data was preprocessed and engineered to create relevant features. The following steps were performed:

Lag Features: A lag of 5 days for the closing price was created to provide past values as features for prediction.
Rolling Statistics: A rolling mean and standard deviation over the past 10 days were added as features to capture trends and volatility.
Normalization: The features were scaled using MinMaxScaler to normalize the data, ensuring that the model can learn efficiently.
Train-Test Split: The dataset was split into 80% training data and 20% testing data.
Neural Network Model
An RNN (Recurrent Neural Network) was chosen for this exercise due to the sequential nature of the stock price data. The architecture consists of the following components:

Input Layer: Takes in the normalized features (lag values, rolling mean, rolling standard deviation).
RNN Layer: A recurrent layer that processes the sequential nature of the data.
Fully Connected Layer: Outputs the predicted closing price.
Hyperparameters
During the training process, the following hyperparameters were experimented with:

Learning Rate: 0.0005
Number of Epochs: 20
Batch Size: 1
Hidden Size: 50 (number of neurons in the RNN layer)
Activation Function: Default ReLU for the fully connected layer.

Training and Evaluation
The model was trained using the Mean Squared Error (MSE) loss function and the Adam optimizer. Training was performed for 20 epochs, with the loss calculated on the training and validation datasets.

After training, the model's performance was evaluated on the test set, and the MSE for the test set was computed. Hyperparameters such as the learning rate and number of epochs were adjusted to see their effect on model performance.

Results
MSE on the Training Set: [Insert result here]
MSE on the Test Set: [Insert result here]
From these results, we observed that adjusting the number of layers, neurons, and learning rate helped improve the model's accuracy in predicting stock prices.

Future Work
Hyperparameter Tuning: Further hyperparameter tuning could be done to improve the model’s performance, such as experimenting with the number of layers, neurons, or different optimizers.
Model Improvement: Other sequential models like LSTMs or GRUs could be explored for better handling of long-term dependencies in the data.
Feature Engineering: Additional features such as technical indicators (e.g., moving averages, RSI) could be incorporated for more accurate predictions.
Requirements
To run the code, install the required dependencies:

Dependencies:
PyTorch
Pandas
NumPy
Scikit-learn
Matplotlib
Yahoo Finance (yfinance)

This README file provides a clear overview of the repository, steps taken in data preprocessing and feature engineering, the neural network architecture used, the training process, and the results obtained. You can customize the content and expand on any part if you have additional details to share.
