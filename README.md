# UPI Fraud Detection using Autoencoders 🔐

This project focuses on detecting fraudulent transactions in UPI (Unified Payments Interface) systems using **unsupervised deep learning techniques**, particularly **Autoencoders**. The model is trained to learn normal transaction patterns and flag anomalies that potentially represent fraud.

> 🚀 **Patent Published Successfully**  
> This project has been successfully submitted and published for **patent** under the guidance of our research team. It focuses on real-time digital payment fraud detection using AI-based systems.

## Problem Statement

With the rise of UPI transactions in India, fraudulent digital transactions have become a major concern. Traditional fraud detection systems rely on rule-based models or supervised learning, which require labeled data (fraud vs. non-fraud). However, due to the rarity of fraud cases, **Autoencoders**, a type of unsupervised neural network, are used to detect outliers in transaction patterns without labeled data.

## Solution Overview

We implemented a **Deep Autoencoder-based Anomaly Detection** model to detect UPI frauds. The model learns the representation of legitimate transactions and detects deviations that could indicate fraud.

## Tech Stack

- **Python 3.x**
- **TensorFlow / Keras** – for building the Autoencoder
- **Pandas, NumPy** – for data handling
- **Matplotlib, Seaborn** – for visualization
- **Streamlit** – to build an interactive frontend UI
- **Jupyter Notebook** – for experimentation and model development

## 📊 Dataset

The dataset simulates real-time UPI transactions with attributes like:
- `Transaction Amount`
- `Merchant ID`
- `Device ID`
- `Transaction Type`
- `Timestamp`
- ...and other behavioral features

*Note: Due to privacy reasons, the dataset is either synthetic or anonymized.*

##  Workflow

1. **Data Preprocessing**
   - Handling missing values
   - Feature scaling
   - Encoding categorical values

2. **Model Training**
   - Building a deep Autoencoder with symmetrical layers
   - Training on normal (non-fraudulent) transactions

3. **Anomaly Detection**
   - Using reconstruction error to flag anomalies
   - Threshold tuning for optimal fraud detection

4. **Streamlit Interface**
   - Upload transaction files
   - Display prediction and anomalies
   - Interactive charts and fraud probability visualization

## How to Run
bash
# Clone the repository
git clone https://github.com/your-username/upi-fraud-detection-autoencoder.git
cd upi-fraud-detection-autoencoder

# Install dependencies
pip install -r requirements.txt

# Run the Streamlit app
streamlit run app.py
