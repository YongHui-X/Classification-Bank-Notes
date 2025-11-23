# Banknote Classification (PyTorch Neural Network)

A simple but effective **binary classifier** that is trained using a neural network to classify whether a banknote is **genuine or forged** using geometric features.  

## Features
- Fully connected neural network (6 → 100 → 50 → 1)
- StandardScaler preprocessing
- Sigmoid binary classification with BCE loss
- Shows test loss, accuracy, and detailed predictions
- Simple, self-contained, easy to run

## Requirements
- torch
- pandas
- scikit-learn
- matplotlib

## Results
- Test Loss (BCE): ~0.0008
- Test Accuracy: ≈100%
<img width="544" height="425" alt="results" src="https://github.com/user-attachments/assets/5550ebbc-e822-4b91-ab12-02f05762f56b" />
<img width="640" height="480" alt="loss curve" src="https://github.com/user-attachments/assets/635f546e-a906-495a-829e-e0a3c7d29473" />
