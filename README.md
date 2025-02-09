# 📊 Neural Network for Time-Series Prediction

![Python](https://img.shields.io/badge/Python-3.8-blue) 
![PyTorch](https://img.shields.io/badge/PyTorch-1.10-red)
![License](https://img.shields.io/badge/License-MIT-green)

This repository contains the code and analysis for building, training, and evaluating neural network models to predict stock prices using historical **NVIDIA (NVDA) stock price data**.

---

## 📌 Overview
This project demonstrates:
✅ **Preprocessing** and feature engineering for time-series data  
✅ Building a **Recurrent Neural Network (RNN)** using PyTorch  
✅ Experimenting with **hyperparameters**  
✅ Evaluating model performance using **Mean Squared Error (MSE)**  
✅ **Insights** and areas for improvement  

---

## 📂 Project Structure
📦 project-folder ├── 📁 data/ # Raw & processed dataset ├── 📁 models/ # PyTorch model definitions & training scripts ├── 📁 notebooks/ # Jupyter notebooks for exploration (if applicable) ├── 📄 README.md # Project documentation ├── 📄 requirements.txt # Required Python packages


---

## 📊 **Dataset**
The dataset consists of historical **NVDA stock prices** from **Yahoo Finance**.

| Feature  | Description |
|----------|------------|
| `Date`   | Date of stock price data |
| `Open`   | Opening price of the stock |
| `High`   | Highest price of the day |
| `Low`    | Lowest price of the day |
| `Close`  | Closing price of the stock (target) |
| `Volume` | Number of shares traded |

🕒 **Data Span:** Last **5 years** of daily trading data.

---

## 🔨 **Feature Engineering**
Before training the model, several preprocessing steps were performed:

- **🔄 Lag Features:** Created **5-day lag** of closing prices to provide past values as inputs.  
- **📈 Rolling Statistics:** Added **10-day rolling mean & standard deviation** to capture trends & volatility.  
- **📊 Normalization:** Applied **MinMaxScaler** to scale the features for better model learning.  
- **✂️ Train-Test Split:** **80% training, 20% testing** data split.

---

## 🏗 **Neural Network Model**
Since stock price data is **sequential**, an **RNN (Recurrent Neural Network)** was implemented.

### 🔹 **Model Architecture**
1️⃣ **Input Layer** → Normalized features (**lag values, rolling mean, rolling std**).  
2️⃣ **RNN Layer** → Processes sequential stock price data.  
3️⃣ **Fully Connected Layer** → Outputs the predicted closing price.

```python
import torch.nn as nn

class RNNModel(nn.Module):
    def __init__(self, input_size, hidden_size=50, output_size=1):
        super(RNNModel, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.rnn(x)
        return self.fc(out[:, -1, :])


🎛 Hyperparameters
During training, different hyperparameters were tested:

Parameter	Value
Learning Rate	0.0005
Number of Epochs	20
Batch Size	1
Hidden Size	50
Activation	ReLU (default for fully connected layer)

