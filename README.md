# 📊 Neural Network for Time-Series Prediction

![Python](https://img.shields.io/badge/Python-3.11-blue) 
![PyTorch](https://img.shields.io/badge/PyTorch-2.5-red)

This repository contains the code and analysis for building, training, and evaluating neural network models to predict stock prices using historical **NVIDIA (NVDA) stock price data**.

---

## 📌 Overview
This project demonstrates:
- ✅ **Preprocessing** and feature engineering for time-series data  
- ✅ Building a **Recurrent Neural Network (RNN)** using PyTorch  
- ✅ Experimenting with **hyperparameters**  
- ✅ Evaluating model performance using **Mean Squared Error (MSE)**  
- ✅ **Insights** and areas for improvement  

---

## 📂 Project Structure
📦 stock-price-prediction ├── 📁 data/ # Raw & processed dataset ├── 📁 models/ # PyTorch model definitions & training scripts ├── 📁 notebooks/ # Jupyter notebooks for exploration (if applicable) ├── 📄 README.md # Project documentation ├── 📄 requirements.txt # Required Python packages

---

## 📊 Dataset
The dataset consists of historical **NVIDIA stock prices** from **Yahoo Finance**.

| Feature          | Description |
|-----------------|-------------|
| `Close Price`   | The closing price of the stock (target variable) |

🕒 **Data Span:** Last **5 years** of daily trading data.

---

## 🔨 Feature Engineering
Before training the model, several preprocessing steps were performed:

- **🔄 Lag Features:** Created **5-day lag** of closing prices to provide past values as inputs.  
- **📈 Rolling Statistics:** Added **10-day rolling mean & standard deviation** to capture trends & volatility.  
- **📊 Normalization:** Applied **MinMaxScaler** to scale the features for better model learning.  
- **✂️ Train-Test Split:** **80% training, 20% testing** data split.

---

## 🏗 Neural Network Model
Since stock price data is **sequential**, an **RNN (Recurrent Neural Network)** was implemented.

### 🔹 Model Architecture
1️⃣ **Input Layer** → Normalized features (**lag values, rolling mean, rolling std**).  
2️⃣ **RNN Layer** → Processes sequential stock price data.  
3️⃣ **Fully Connected Layer** → Outputs the predicted closing price.

### 📜 Code: RNN Model
```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset

# RNN Model Definition
class RNNModel(nn.Module):
    def __init__(self, input_size, hidden_size=50, output_size=1):
        super(RNNModel, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.rnn(x)
        return self.fc(out[:, -1, :])

# Load dataset
df = pd.read_csv('data/NVDA_stock.csv')

# Feature Engineering
df['Close_Lag_5'] = df['Close'].shift(5)
df['Rolling_Mean_10'] = df['Close'].rolling(window=10).mean()
df['Rolling_Std_10'] = df['Close'].rolling(window=10).std()
df.dropna(inplace=True)

# Normalize features
scaler = MinMaxScaler()
features = ['Close_Lag_5', 'Rolling_Mean_10', 'Rolling_Std_10']
df[features] = scaler.fit_transform(df[features])

# Prepare dataset
X = df[features].values
y = df['Close'].values
X = np.expand_dims(X, axis=1)  # Reshape for RNN

# Split dataset
split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

# DataLoader
train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), batch_size=1, shuffle=True)
test_loader = DataLoader(TensorDataset(X_test_tensor, y_test_tensor), batch_size=1)

# Initialize model
model = RNNModel(input_size=3)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0005)

# Train model
epochs = 20
for epoch in range(epochs):
    for batch_x, batch_y in train_loader:
        optimizer.zero_grad()
        y_pred = model(batch_x)
        loss = criterion(y_pred, batch_y)
        loss.backward()
        optimizer.step()

# Evaluate model
with torch.no_grad():
    test_predictions = model(X_test_tensor)
    test_loss = criterion(test_predictions, y_test_tensor)
    print(f'Test MSE: {test_loss.item()}')
```

## 🎛 Hyperparameters
During training, different hyperparameters were tested:

- Parameter	Value
- Learning Rate	0.001 & 0.0005
- Number of Epochs	20 & 100
- Batch Size	1
- Hidden Size	50
- Activation	ReLU (default for fully connected layer)

## 📈 Training & Evaluation
The model was trained using:

- Loss Function: Mean Squared Error (MSE)
- Optimizer: Adam

After training, the model was evaluated on the test set, and the performance was measured using MSE and visualization 

## ⚙️ Installation & Setup
To run the project, install the required dependencies:

``` python
git clone https://github.com/UlrikN1234/BDS-2025.git
cd BDS-2025
pip install -r requirements.txt
```

📦 Dependencies
- PyTorch
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Yahoo Finance (yfinance)

🔗 Contributions are welcome! Feel free to fork and improve the model.
