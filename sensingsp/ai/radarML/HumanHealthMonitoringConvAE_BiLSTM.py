import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import pywt
import pandas as pd
from pathlib import Path

# Custom Dataset Class
class RadarECGDataset(Dataset):
    def __init__(self, radar_signals, ecg_signals):
        self.radar_signals = radar_signals
        self.ecg_signals = ecg_signals

    def __len__(self):
        return len(self.radar_signals)

    def __getitem__(self, idx):
        return torch.tensor(self.radar_signals[idx], dtype=torch.float32), torch.tensor(self.ecg_signals[idx], dtype=torch.float32)

# Convolutional Autoencoder and BiLSTM Model
class ConvAutoencoderBiLSTM(nn.Module):
    def __init__(self):
        super(ConvAutoencoderBiLSTM, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=3, kernel_size=4, padding='same'),
            nn.LayerNorm(1024),
            nn.Dropout(0.2),
            nn.Conv1d(3, 64, kernel_size=8, stride=8, padding='same'),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.2),
            nn.MaxPool1d(2, padding='same'),
            nn.Conv1d(64, 32, kernel_size=8, stride=4, padding='same'),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.MaxPool1d(2, padding='same')
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(32, 32, kernel_size=8, stride=4, padding='same'),
            nn.ReLU(),
            nn.ConvTranspose1d(32, 64, kernel_size=8, stride=8, padding='same'),
            nn.BatchNorm1d(64)
        )
        self.bilstm = nn.LSTM(input_size=64, hidden_size=8, num_layers=1, batch_first=True, bidirectional=True)
        self.fc = nn.Sequential(
            nn.Linear(16, 4),
            nn.Linear(4, 2),
            nn.Linear(2, 1)
        )

    def forward(self, x):
        x = x.unsqueeze(1)  # Add channel dimension
        x = self.encoder(x)
        x = self.decoder(x)
        x = x.permute(0, 2, 1)  # Change to (batch, time, features) for LSTM
        x, _ = self.bilstm(x)
        x = self.fc(x)
        return x.squeeze(-1)

# Helper function for MODWT using PyWavelets
def modwt(signal, wavelet='sym4', level=5):
    coeffs = pywt.wavedec(signal, wavelet, level=level, mode='symmetric')
    return coeffs

# Data Preparation
def prepare_data(file_path):
    # Assuming file_path points to npz files containing 'radar' and 'ecg' arrays
    data = np.load(file_path)
    radar_signals = data['radar']
    ecg_signals = data['ecg']

    # Normalize ECG signals
    scaler = MinMaxScaler()
    ecg_signals = scaler.fit_transform(ecg_signals)

    return radar_signals, ecg_signals

# Define dataset folder paths
dataset_folder = "C:/Users/moein.ahmadi/OneDrive - University of Luxembourg/ThingsToDo/RadarAIDatasets/SynchronizedRadarECGData"
radar_train_val_folder = os.path.join(dataset_folder, "trainVal", "radar")
radar_test_folder = os.path.join(dataset_folder, "test", "radar")
ecg_train_val_folder = os.path.join(dataset_folder, "trainVal", "ecg")
ecg_test_folder = os.path.join(dataset_folder, "test", "ecg")

# Function to extract categories from filenames
def filenames_to_labels(folder_path, extract_before="_radar"):
    filenames = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    labels = [Path(f).stem.split(extract_before)[0] for f in filenames]
    return pd.Series(labels, name="Categories")

# Extract categories and compute their distribution for train and test datasets
radar_train_labels = filenames_to_labels(radar_train_val_folder, extract_before="_radar")
radar_test_labels = filenames_to_labels(radar_test_folder, extract_before="_radar")

# Display the summary (category distribution)
print("Radar Train Categories Summary:")
print(radar_train_labels.value_counts())

print("\nRadar Test Categories Summary:")
print(radar_test_labels.value_counts())

for label in radar_train_labels:
    print(label)
# Load Data
train_file = "trainValData.npz"
radar_train, ecg_train = prepare_data(train_file)
dataset = RadarECGDataset(radar_train, ecg_train)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

# Model Training
model = ConvAutoencoderBiLSTM()
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

num_epochs = 20

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for radar, ecg in dataloader:
        radar, ecg = radar.to(device), ecg.to(device)
        optimizer.zero_grad()
        outputs = model(radar)
        loss = criterion(outputs, ecg)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * radar.size(0)

    epoch_loss = running_loss / len(dataloader.dataset)
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}")

# Model Evaluation
def evaluate_model(model, dataloader, device):
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for radar, ecg in dataloader:
            radar, ecg = radar.to(device), ecg.to(device)
            preds = model(radar)
            all_preds.append(preds.cpu().numpy())
            all_targets.append(ecg.cpu().numpy())

    return np.concatenate(all_preds), np.concatenate(all_targets)

# Evaluate Model
val_file = "testData.npz"
radar_val, ecg_val = prepare_data(val_file)
val_dataset = RadarECGDataset(radar_val, ecg_val)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
preds, targets = evaluate_model(model, val_loader, device)

# Plotting Predicted vs. Actual ECG Signals
plt.figure(figsize=(12, 8))
plt.plot(preds[0], label='Predicted ECG')
plt.plot(targets[0], label='Actual ECG', linestyle='--')
plt.xlabel('Time Steps')
plt.ylabel('ECG Signal')
plt.title('Predicted vs Actual ECG Signals')
plt.legend()
plt.grid()
plt.show()
