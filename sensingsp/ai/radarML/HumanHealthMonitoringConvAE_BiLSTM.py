def runradarConvAEBiLSTMapp():

    import os
    import glob
    import requests
    import zipfile
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader, random_split
    from scipy.io import loadmat
    import matplotlib.pyplot as plt
    import sensingsp as ssp
    import torch.nn.functional as F

# trainCats: 830×1 categorical

#      GDN0001_Resting        59 
#      GDN0001_Valsalva       97 
#      GDN0002_Resting        60 
#      GDN0002_Valsalva       97 
#      GDN0003_Resting        58 
#      GDN0003_Valsalva      103 
#      GDN0004_Apnea          14 
#      GDN0004_Resting        58 
#      GDN0004_Valsalva      106 
#      GDN0005_Apnea          14 
#      GDN0005_Resting        59 
#      GDN0005_Valsalva      105 
#      <undefined>             0 
# testCats: 200×1 categorical

#      GDN0006_Apnea          14 
#      GDN0006_Resting        59 
#      GDN0006_Valsalva      127 
#      <undefined>             0 

    class MODWTLayer(nn.Module):
        def __init__(self, wavelet="sym4", level=5, include_lowpass=True,selected_Levels=[3,4,5],aggregate_levels=False, boundary="periodic", name="MODWT"):
            super(MODWTLayer, self).__init__()

            self.wavelet = wavelet
            self.level = level
            self.include_lowpass = include_lowpass
            self.aggregate_levels = aggregate_levels
            self.boundary = boundary
            self.selected_Levels = selected_Levels
            self.name = name

            # Get wavelet filters
            self.lowpass_filter, self.highpass_filter = self.get_wavelet_filters(wavelet)
            self.min_length = 2 ** self.level

        def forward(self, x):
            """
            x: Tensor of shape (B, C, T), where B is batch size, C is channels, T is the time dimension.
            Returns: Tensor of shape (B, C, S, T), where S is the spatial (level) dimension.
            """
            if x.size(-1) < self.min_length:
                raise ValueError(f"Input length must be at least {self.min_length} for level {self.level}.")

            # Apply MODWT
            coeffs = self._modwt(x)

            if self.include_lowpass:
                coeffs.append(coeffs.pop(-1))  # Move smooth coefficients to the end

            # Stack coefficients along the spatial dimension
            # result = torch.stack(coeffs, dim=2)  # (B, C, S, T)
            result = torch.stack([coeffs[i] for i in self.selected_Levels], dim=2)
            if self.aggregate_levels:
                result = result.sum(dim=2, keepdim=True)  # Aggregate levels

            return result

        def _modwt(self, x):
            """Compute MODWT for the input tensor."""
            coeffs = []
            current_signal = x

            for level in range(self.level):
                lowpass_output = self._convolve(current_signal, self.lowpass_filter, self.boundary)
                highpass_output = self._convolve(current_signal, self.highpass_filter, self.boundary)

                # Trim outputs to match input size
                min_len = min(lowpass_output.size(-1), x.size(-1))
                lowpass_output = lowpass_output[..., :min_len]
                highpass_output = highpass_output[..., :min_len]

                coeffs.append(highpass_output)  # Add detail coefficients
                current_signal = lowpass_output  # Continue with lowpass for next level

            # Trim final smooth coefficients to match input size
            min_len = min(current_signal.size(-1), x.size(-1))
            current_signal = current_signal[..., :min_len]

            coeffs.append(current_signal)  # Add final smooth coefficients
            return coeffs


        def _convolve(self, x, filter_coeffs, boundary):
            """Apply convolution with the specified boundary handling."""
            if boundary == "periodic":
                padding = len(filter_coeffs) // 2
                x = torch.nn.functional.pad(x, (padding, padding), mode="circular")
            elif boundary == "reflection":
                padding = len(filter_coeffs) // 2
                x = torch.nn.functional.pad(x, (padding, padding), mode="reflect")
            else:
                raise ValueError(f"Unsupported boundary type: {boundary}")

            filter_tensor = torch.tensor(filter_coeffs, dtype=x.dtype, device=x.device).unsqueeze(0).unsqueeze(0)
            return torch.nn.functional.conv1d(x, filter_tensor, groups=x.size(1))

        def get_wavelet_filters(self, wavelet_name):
            """
            Returns the low-pass (dec_lo) and high-pass (dec_hi) decomposition filters
            for the specified wavelet without using PyWavelets.
            
            Args:
                wavelet_name (str): Name of the wavelet, such as:
                    - "haar" (equivalent to "db1")
                    - "db2", "db3", ... (Daubechies)
                    - "sym2", "sym3", "sym4", ... (Symlets)

            Returns:
                tuple (dec_lo, dec_hi): low-pass and high-pass decomposition filters as lists.
            """
            wavelets = {
                "haar": {
                    "dec_lo": [0.7071067811865476, 0.7071067811865476],
                    "dec_hi": [-0.7071067811865476, 0.7071067811865476],
                },
                "db1": {
                    "dec_lo": [0.7071067811865476, 0.7071067811865476],
                    "dec_hi": [-0.7071067811865476, 0.7071067811865476],
                },
                "db2": {
                    "dec_lo": [
                        -0.12940952255092145, 0.22414386804185735,
                        0.8365163037378079, 0.48296291314469025
                    ],
                    "dec_hi": [
                        -0.48296291314469025, 0.8365163037378079,
                        -0.22414386804185735, -0.12940952255092145
                    ],
                },
                "db3": {
                    "dec_lo": [
                        0.035226291885709536, -0.08544127388224149,
                        -0.13501102001025458, 0.4598775021193313,
                        0.8068915093133388, 0.3326705529509569
                    ],
                    "dec_hi": [
                        -0.3326705529509569, 0.8068915093133388,
                        -0.4598775021193313, -0.13501102001025458,
                        0.08544127388224149, 0.035226291885709536
                    ],
                },
                "db4": {
                    "dec_lo": [
                        -0.0105974017850021, 0.032883011666982945,
                        0.030841381835986965, -0.18703481171888114,
                        -0.02798376941698385, 0.6308807679295904,
                        0.7148465705529154, 0.23037781330885523
                    ],
                    "dec_hi": [
                        -0.23037781330885523, 0.7148465705529154,
                        -0.6308807679295904, -0.02798376941698385,
                        0.18703481171888114, 0.030841381835986965,
                        -0.032883011666982945, -0.0105974017850021
                    ],
                },
                "sym2": {
                    "dec_lo": [
                        -0.4829629131445341, 0.8365163037378077,
                        0.22414386804185735, -0.12940952255126034
                    ],
                    "dec_hi": [
                        0.12940952255126034, 0.22414386804185735,
                        -0.8365163037378077, -0.4829629131445341
                    ],
                },
                "sym3": {
                    "dec_lo": [
                        0.019538882735286728, -0.021101834024758855,
                        -0.17532808990845047, 0.01660210576452232,
                        0.6339789634582119, 0.7234076904024206,
                        0.1993975339773936, -0.039134249302383094
                    ],
                    "dec_hi": [
                        0.039134249302383094, 0.1993975339773936,
                        -0.7234076904024206, 0.6339789634582119,
                        -0.01660210576452232, -0.17532808990845047,
                        0.021101834024758855, 0.019538882735286728
                    ],
                },
                "sym4": {
                    "dec_lo": [
                        -0.07576571478927333, -0.02963552764599851,
                        0.49761866763201545, 0.8037387518059161,
                        0.29785779560527736, -0.09921954357684722,
                        -0.012603967262037833, 0.032223100604071306
                    ],
                    "dec_hi": [
                        -0.032223100604071306, -0.012603967262037833,
                        0.09921954357684722, 0.29785779560527736,
                        -0.8037387518059161, 0.49761866763201545,
                        0.02963552764599851, -0.07576571478927333
                    ],
                }
            }

            # Check if requested wavelet is available
            if wavelet_name not in wavelets:
                raise ValueError(f"Wavelet '{wavelet_name}' is not supported in the current dictionary.")

            dec_lo = wavelets[wavelet_name]["dec_lo"]
            dec_hi = wavelets[wavelet_name]["dec_hi"]
            return dec_lo, dec_hi
        
    if 0:
        # Example usage equivalent to MATLAB
        # Create MODWT layer with specified parameters
        layer = MODWTLayer(wavelet="sym4", level=5, include_lowpass=False, aggregate_levels=False)

        # Create a 1x1x64 input tensor: 1 channel, batch size 1, length 64
        x = torch.randn(1, 1, 64)

        # Run the forward pass through the MODWT layer
        output = layer(x)

        # Select levels 3 to 5 from the output
        selected_levels = [2, 3, 4]  # Python is zero-indexed, so level 3 maps to index 2
        output_selected = output[:, :, selected_levels, :]

        print("Output shape:", output_selected.shape)
        # Output shape: torch.Size([1, 1, 3, 64])


    # Download and Extract the Data
    # ==============================
    zip_path = "SynchronizedRadarECGData.zip"
    url = "https://ssd.mathworks.com/supportfiles/SPT/data/SynchronizedRadarECGData.zip"
    zip_folder = os.path.join(ssp.config.temp_folder, "datasets", "SynchronizedRadarECGData")

    # Check if the folder exists
    if not os.path.exists(zip_folder):

        # Download the ZIP file
        print("Downloading the ZIP file...")
        response = requests.get(url, stream=True)
        zip_path = os.path.join(ssp.config.temp_folder, "datasets", "SynchronizedRadarECGData.zip")

        # Save the ZIP file locally
        with open(zip_path, "wb") as f:
            f.write(response.content)
        print("Download complete!")

        # Extract the ZIP file
        print("Extracting the ZIP file...")
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(zip_folder)
        print(f"Data extracted to {zip_folder}")

        # Clean up: Remove the ZIP file
        os.remove(zip_path)
        print("Temporary ZIP file removed.")

    else:
        print(f"The folder {zip_folder} already exists. No action taken.")
    
    # ==============================
    # Device Setup
    # ==============================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # ==============================
    # Normalization and Dataset
    # ==============================
    def normalize_ecg(ecg_signal):
        ecg_signal = ecg_signal - np.median(ecg_signal)
        max_val = np.max(np.abs(ecg_signal))
        if max_val < 1e-12:
            max_val = 1.0
        ecg_signal = ecg_signal / max_val
        return ecg_signal

    class RadarECGDataset(Dataset):
        def __init__(self, radar_dir, ecg_dir, transform_ecg=True):
            self.radar_files = sorted(glob.glob(os.path.join(radar_dir, '*.mat')))
            self.ecg_files = sorted(glob.glob(os.path.join(ecg_dir, '*.mat')))
            assert len(self.radar_files) == len(self.ecg_files), "Mismatch in number of radar and ECG files."
            self.transform_ecg = transform_ecg

        def __len__(self):
            return len(self.radar_files)

        def __getitem__(self, idx):
            radar_mat = loadmat(self.radar_files[idx])
            ecg_mat = loadmat(self.ecg_files[idx])

            # Adjust the key if necessary. Assuming 'signal' is correct.
            radar_signal = radar_mat['radar_l'].squeeze() # (1024,)
            ecg_signal = ecg_mat['ecg_l'].squeeze() # (1024,)

            if self.transform_ecg:
                ecg_signal = normalize_ecg(ecg_signal)

            radar_signal = torch.tensor(radar_signal, dtype=torch.float32).unsqueeze(0)  # (1, 1024)
            ecg_signal = torch.tensor(ecg_signal, dtype=torch.float32).unsqueeze(0)      # (1, 1024)

            return radar_signal, ecg_signal

    # ==============================
    # Create Datasets and Loaders
    # ==============================
    dataset_folder = zip_folder
    trainVal_radar_dir = os.path.join(dataset_folder, "trainVal", "radar")
    trainVal_ecg_dir = os.path.join(dataset_folder, "trainVal", "ecg")
    test_radar_dir = os.path.join(dataset_folder, "test", "radar")
    test_ecg_dir = os.path.join(dataset_folder, "test", "ecg")

    trainVal_dataset = RadarECGDataset(trainVal_radar_dir, trainVal_ecg_dir, transform_ecg=True)
    test_dataset = RadarECGDataset(test_radar_dir, test_ecg_dir, transform_ecg=True)

    train_size = int(0.85 * len(trainVal_dataset))
    val_size = len(trainVal_dataset) - train_size
    train_dataset, val_dataset = random_split(trainVal_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    # ==============================
    # Model Definition
    # ==============================
    class ECGReconstructionModel(nn.Module):
        def __init__(self):
            super(ECGReconstructionModel, self).__init__()
            
            # # MODWT layer for initial wavelet decomposition
            # self.modwt_layer = MODWTLayer(
            #     wavelet="sym4",  # Wavelet type
            #     level=5,         # Number of decomposition levels
            #     include_lowpass=False,  # Exclude lowpass coefficients
            # )
            
            # # Normalization and dropout after MODWT
            # self.layernorm = nn.LayerNorm([3, 1024])  # Adjusted normalization dimensions
            # self.dropout1 = nn.Dropout(0.2)
            
            self.sequence_input = nn.Identity()  # Sequence input placeholder
            self.modwt = MODWTLayer(wavelet="sym4", level=5, include_lowpass=False)
            self.flatten = nn.Flatten(start_dim=1)
            self.layer_norm = nn.LayerNorm(3072)  # Replace with appropriate dimension
            self.dropout1 = nn.Dropout(0.2)
            
            # First convolution block
            self.conv1 = nn.Conv1d(in_channels=1, out_channels=8, kernel_size=64, padding=3, stride=8)
            self.relu1 = nn.ReLU()
            self.bn1 = nn.BatchNorm1d(64)
            self.dropout2 = nn.Dropout(0.2)
            self.pool1 = nn.MaxPool1d(kernel_size=2, padding=1)
            
            # Second convolution block
            self.conv2 = nn.Conv1d(in_channels=64, out_channels=32, kernel_size=8, padding=3, stride=4)
            self.relu2 = nn.ReLU()
            self.bn2 = nn.BatchNorm1d(32)
            self.pool2 = nn.MaxPool1d(kernel_size=2, padding=1)
            
            # Transposed convolution layers for upsampling
            self.tconv1 = nn.ConvTranspose1d(in_channels=32, out_channels=32, kernel_size=8, stride=4, padding=3)
            self.relu3 = nn.ReLU()
            self.tconv2 = nn.ConvTranspose1d(in_channels=32, out_channels=64, kernel_size=8, stride=8, padding=3)
            
            # LSTM for sequence modeling
            self.lstm = nn.LSTM(input_size=64, hidden_size=8, bidirectional=True, batch_first=True)
            
            # Fully connected layers for final reconstruction
            self.fc1 = nn.Linear(16, 4)  # LSTM output size: 8 * 2 (bidirectional)
            self.fc2 = nn.Linear(4, 2)
            self.fc3 = nn.Linear(2, 1)
        
        def forward(self, x):
            # # Apply MODWT
            # coeffs = self.modwt_layer(x)  # MODWT produces detail coefficients for levels 3, 4, 5
            # # Layer normalization and dropout
            # x = self.layernorm(x) 
            # x = self.dropout1(x)
                
            # print(x.shape) # ([64, 1, 1024])
            x = self.sequence_input(x)
            # print(x.shape) # ([64, 1, 1024])
            x = self.modwt(x)
            # print(x.shape) # ([64, 1, 3, 1024])
            x = self.flatten(x)
            # print(x.shape) # torch.Size([64, 3072])
            x = self.layer_norm(x) # Given normalized_shape=[64], expected input with shape [*, 64], but got input of size[64, 6144]
            # print(x.shape)# torch.Size([64, 3072])
            x = self.dropout1(x)
            # print(x.shape)# torch.Size([64, 3072])
            
            # First convolution block
            x = self.conv1(x)
            print(x.shape)
            x = self.relu1(x)
            print(x.shape)
            x = self.bn1(x)
            print(x.shape)
            x = self.dropout2(x)
            x = self.pool1(x)
            print(x.shape)
            # Second convolution block
            x = self.conv2(x)
            x = self.relu2(x)
            x = self.bn2(x)
            x = self.pool2(x)
            
            # Transposed convolutions for upsampling
            x = self.tconv1(x)
            x = self.relu3(x)
            x = self.tconv2(x)
            
            # Prepare for LSTM
            x = x.transpose(1, 2)  # Change shape to (batch_size, sequence_length, channels)
            
            # LSTM for sequence modeling
            output, _ = self.lstm(x)  # Output shape: (batch_size, sequence_length, hidden_size * 2)
            
            # Fully connected layers
            x = self.fc1(output)
            x = self.fc2(x)
            x = self.fc3(x)
            
            # Return to original shape (batch_size, channels, sequence_length)
            x = x.transpose(1, 2)
            return x

    # ==============================
    # Training Setup
    # ==============================
    model = ECGReconstructionModel().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    max_epochs = 5

    # ==============================
    # Training Loop
    # ==============================
    for epoch in range(max_epochs):
        model.train()
        train_loss = 0.0
        for radar_batch, ecg_batch in train_loader:
            radar_batch = radar_batch.to(device)
            ecg_batch = ecg_batch.to(device)

            optimizer.zero_grad()
            outputs = model(radar_batch)
            loss = criterion(outputs, ecg_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * radar_batch.size(0)
        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for radar_batch, ecg_batch in val_loader:
                radar_batch = radar_batch.to(device)
                ecg_batch = ecg_batch.to(device)
                outputs = model(radar_batch)
                loss = criterion(outputs, ecg_batch)
                val_loss += loss.item() * radar_batch.size(0)
        val_loss /= len(val_loader.dataset)

        print(f"Epoch [{epoch+1}/{max_epochs}] Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

    # ==============================
    # Testing
    # ==============================
    model.eval()
    test_losses = []
    with torch.no_grad():
        for radar_batch, ecg_batch in test_loader:
            radar_batch = radar_batch.to(device)
            ecg_batch = ecg_batch.to(device)
            outputs = model(radar_batch)
            loss = criterion(outputs, ecg_batch)
            test_losses.append(loss.item())

    test_loss_mean = np.mean(test_losses)
    print("Test Loss:", test_loss_mean)

    # ==============================
    # Visualization
    # ==============================
    model.eval()
    with torch.no_grad():
        radar_batch, ecg_batch = next(iter(test_loader))
        radar_batch = radar_batch.to(device)
        outputs = model(radar_batch)
        outputs = outputs.squeeze().cpu().numpy()
        ecg_ref = ecg_batch.squeeze().numpy()

    plt.figure(figsize=(10,4))
    plt.plot(ecg_ref, label='Measured ECG')
    plt.plot(outputs, label='Reconstructed ECG')
    plt.title("ECG Reconstruction Sample")
    plt.legend()
    plt.grid(True)
    plt.show()


# ----------------------------
# Main Execution
# ----------------------------
if __name__ == "__main__":
    runradarConvAEBiLSTMapp()







# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# # Define the custom modwtLayer
# class ModwtLayer(nn.Module):
#     def __init__(self, weights_shape):
#         super(ModwtLayer, self).__init__()
#         self.weights = nn.Parameter(torch.zeros(weights_shape), requires_grad=True)

#     def forward(self, x):
#         # Implementing a simple forward pass using weights
#         batch_size, seq_len, channels = x.shape
#         weights = self.weights.unsqueeze(0).repeat(batch_size, 1, 1)  # Expand weights for batch processing
#         output = torch.matmul(x, weights)  # Example operation (modify as needed)
#         return output

# # Define the PyTorch model
# class MyModel(nn.Module):
#     def __init__(self):
#         super(MyModel, self).__init__()

#         self.modwt_layer = ModwtLayer((1, 8))
#         self.layer_norm = nn.LayerNorm(8, eps=1e-5)
#         self.dropout1 = nn.Dropout(0.2)
#         self.conv1d_1 = nn.Conv1d(in_channels=8, out_channels=8, kernel_size=64, stride=8, padding=32)
#         self.batchnorm1 = nn.BatchNorm1d(num_features=8, eps=1e-5)
#         self.maxpool1d_1 = nn.MaxPool1d(kernel_size=2, stride=1, padding=1)

#         self.conv1d_2 = nn.Conv1d(in_channels=8, out_channels=8, kernel_size=32, stride=4, padding=16)
#         self.batchnorm2 = nn.BatchNorm1d(num_features=8, eps=1e-5)
#         self.maxpool1d_2 = nn.MaxPool1d(kernel_size=2, stride=1, padding=1)

#         self.transposed_conv1d_1 = nn.ConvTranspose1d(in_channels=8, out_channels=8, kernel_size=32, stride=4, padding=16)
#         self.transposed_conv1d_2 = nn.ConvTranspose1d(in_channels=8, out_channels=8, kernel_size=64, stride=8, padding=32)

#         self.bilstm = nn.LSTM(input_size=8, hidden_size=8, num_layers=1, bidirectional=True, batch_first=True)

#         self.fc1 = nn.Linear(16, 4)  # Bidirectional LSTM doubles the feature size
#         self.fc2 = nn.Linear(4, 2)
#         self.fc3 = nn.Linear(2, 1)

#     def forward(self, x):
#         x = self.modwt_layer(x)
#         x = x.permute(0, 2, 1)  # Equivalent to TimeDistributed(Permute(2,1))
#         x = x.flatten(start_dim=2)  # Equivalent to TimeDistributed(Flatten())
#         x = self.layer_norm(x)
#         x = self.dropout1(x)

#         x = x.transpose(1, 2)  # Switch to (B, C, T) for Conv1d
#         x = self.conv1d_1(x)
#         x = F.relu(x)
#         x = self.batchnorm1(x)
#         x = self.maxpool1d_1(x)

#         x = self.conv1d_2(x)
#         x = F.relu(x)
#         x = self.batchnorm2(x)
#         x = self.maxpool1d_2(x)

#         x = self.transposed_conv1d_1(x)
#         x = F.relu(x)
#         x = self.transposed_conv1d_2(x)

#         x = x.transpose(1, 2)  # Back to (B, T, C) for LSTM
#         x, _ = self.bilstm(x)

#         x = self.fc1(x)
#         x = self.fc2(x)
#         x = self.fc3(x)
#         return x

# # Instantiate the model
# model = MyModel()
# print(model)



# import tensorflow as tf
# from tensorflow import keras
# from tensorflow.keras import layers
# from myModel.customLayers.modwtLayer import modwtLayer

# def create_model():
#     sequenceinput = keras.Input(shape=(None,1))
#     layer = modwtLayer(name="layer_", Weights_Shape_=(2,8))(sequenceinput)
#     layerperm = layers.TimeDistributed(layers.Permute((2,1)))(layer)
#     flatten = layers.TimeDistributed(layers.Flatten())(layerperm)
#     layernorm = layers.LayerNormalization(axis=-1, epsilon=0.000010, name="layernorm_")(flatten)
#     dropout_1 = layers.Dropout(0.200000)(layernorm)
#     conv1d_1 = layers.Conv1D(8, 64, strides=(8), padding="same", name="conv1d_1_")(dropout_1)
#     relu_1 = layers.ReLU()(conv1d_1)
#     batchnorm_1 = layers.BatchNormalization(epsilon=0.000010, name="batchnorm_1_")(relu_1)
#     dropout_2 = layers.Dropout(0.200000)(batchnorm_1)
#     maxpool1d_1 = layers.MaxPool1D(pool_size=2, strides=1, padding="same")(dropout_2)
#     conv1d_2 = layers.Conv1D(8, 32, strides=(4), padding="same", name="conv1d_2_")(maxpool1d_1)
#     relu_2 = layers.ReLU()(conv1d_2)
#     batchnorm_2 = layers.BatchNormalization(epsilon=0.000010, name="batchnorm_2_")(relu_2)
#     maxpool1d_2 = layers.MaxPool1D(pool_size=2, strides=1, padding="same")(batchnorm_2)
#     transposed_conv1d_1 = layers.Conv1DTranspose(8, 32, strides=4, padding="same", name="transposed_conv1d_1_")(maxpool1d_2)
#     relu_3 = layers.ReLU()(transposed_conv1d_1)
#     transposed_conv1d_2 = layers.Conv1DTranspose(8, 64, strides=8, padding="same", name="transposed_conv1d_2_")(relu_3)
#     transposed_conv1d_2_bilstm_input = transposed_conv1d_2
#     biLSTM = layers.Bidirectional(layers.LSTM(8, activation='tanh', recurrent_activation='sigmoid', return_sequences=True, return_state=False), name="biLSTM_")(transposed_conv1d_2_bilstm_input)
#     fc_1 = layers.Dense(4, name="fc_1_")(biLSTM)
#     fc_2 = layers.Dense(2, name="fc_2_")(fc_1)
#     fc_3 = layers.Dense(1, name="fc_3_")(fc_2)

#     model = keras.Model(inputs=[sequenceinput], outputs=[fc_3])
#     return model