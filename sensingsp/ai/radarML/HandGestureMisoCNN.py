import os
import re
import sys
import numpy as np
import sensingsp as ssp
import scipy.io
from matplotlib import pyplot as plt
from scipy.signal import lfilter

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split

# Add these imports for confusion matrix
from sklearn.metrics import confusion_matrix
import seaborn as sns

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget,QDialog, QVBoxLayout, QFormLayout, QLineEdit, QLabel, QPushButton, QHBoxLayout, QScrollArea, QFileDialog, QMessageBox, QSpinBox, QDoubleSpinBox, QCheckBox, QComboBox
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap
from torch.utils.tensorboard import SummaryWriter

# Dataset Preparation
class RadarGestureDataset(Dataset):
    def __init__(self, data_folder,max_folder_number=1e6,clutter_removal=True):
        self.data_folder = data_folder
        self.folders_data_classes = []  # Stores data and gesture numbers
        self.classes = {}  # Maps gesture numbers to class indices
        self.data_samples_per_file = 100
        self.slow_time_per_sample = 90
        self.clutter_removal = clutter_removal
        self.max_folder_number = max_folder_number
        NValid = self.data_samples_per_file * self.slow_time_per_sample

        class_idx = 0
        folder_number = 0
                
        for subject_folder in sorted(os.listdir(data_folder)):
            folder_number +=1
            if folder_number > self.max_folder_number:
                break
                
            subject_path = os.path.join(data_folder, subject_folder)
            if os.path.isdir(subject_path):
                for mat_file in sorted(os.listdir(subject_path)):
                    if mat_file.endswith(".mat"):
                        if len(self.folders_data_classes)>100000:
                            continue
                        file_path = os.path.join(subject_path, mat_file)
                        gesture_name = mat_file.split("_")[2]
                        mat_data = scipy.io.loadmat(file_path)

                        left = mat_data.get("Left")
                        top = mat_data.get("Top")
                        right = mat_data.get("Right")

                        if left.shape[0] != NValid:
                            continue

                        if self.clutter_removal:
                            b = [1, -1]
                            a = [1, -0.9]
                            left = lfilter(b, a, left, axis=0)
                            top = lfilter(b, a, top, axis=0)
                            right = lfilter(b, a, right, axis=0)

                        gesture_number = int(re.sub(r'\D', '', gesture_name))
                        if gesture_number not in self.classes:
                            self.classes[gesture_number] = class_idx
                            class_idx += 1

                        self.folders_data_classes.append([left, top, right, gesture_number])

    def __len__(self):
        return len(self.folders_data_classes) * self.data_samples_per_file

    def __getitem__(self, idx):
        file_idx = idx // self.data_samples_per_file
        sample_idx = idx % self.data_samples_per_file
        left, top, right, gesture_number = self.folders_data_classes[file_idx]

        start_idx = sample_idx * self.slow_time_per_sample
        end_idx = start_idx + self.slow_time_per_sample

        left_sample = left[start_idx:end_idx, :]
        top_sample = top[start_idx:end_idx, :]
        right_sample = right[start_idx:end_idx, :]

        # Ensure there are no empty dimensions
        if (left_sample.shape[0] == 0 or
                top_sample.shape[0] == 0 or
                right_sample.shape[0] == 0):
            raise ValueError("Sample data has an empty dimension.")

        radar_tensor = torch.tensor([left_sample, top_sample, right_sample], dtype=torch.float32)
        label = torch.tensor(self.classes[gesture_number], dtype=torch.long)
        return radar_tensor, label

# Model Definition
class RepeatBranch(nn.Module):
    def __init__(self):
        super(RepeatBranch, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

    def forward(self, x):
        return self.layers(x)

class MainBranch(nn.Module):
    def __init__(self, num_classes=12):
        super(MainBranch, self).__init__()
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        return self.fc(x)

class MisoCNN(nn.Module):
    def __init__(self, num_classes=12):
        super(MisoCNN, self).__init__()
        self.branch1 = RepeatBranch()
        self.branch2 = RepeatBranch()
        self.branch3 = RepeatBranch()
        self.main_branch = MainBranch(num_classes)

    def forward(self, x1, x2, x3):
        out1 = self.branch1(x1)
        out2 = self.branch2(x2)
        out3 = self.branch3(x3)

        out1 = nn.functional.adaptive_avg_pool2d(out1, (1, 1)).view(out1.size(0), -1)
        out2 = nn.functional.adaptive_avg_pool2d(out2, (1, 1)).view(out2.size(0), -1)
        out3 = nn.functional.adaptive_avg_pool2d(out3, (1, 1)).view(out3.size(0), -1)

        added_out = out1 + out2 + out3
        final_output = self.main_branch(added_out)
        return final_output

# Training Function
def train_model(model, train_loader, val_loader, num_epochs, criterion, optimizer, device):
    model.to(device)
    fig, ax = plt.subplots(2, 1)
    epoch_loss_plot = []
    for epoch in range(num_epochs):
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        loss_plot = []
        for radar_data, labels in train_loader:
            radar_data, labels = radar_data.to(device), labels.to(device)
            x1 = radar_data[:, 0:1, :, :]
            x2 = radar_data[:, 1:2, :, :]
            x3 = radar_data[:, 2:3, :, :]

            optimizer.zero_grad()
            outputs = model(x1, x2, x3)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            loss_plot.append(100. * correct / total)
            ax[0].clear()
            ax[0].plot(loss_plot)
            ax[0].set_title(f"Epoch {epoch + 1}")
            ax[0].set_xlabel("Iteration")
            ax[0].set_ylabel("Accuracy")
            ax[0].grid(True) 
            plt.draw()
            plt.pause(0.001)
            plt.gcf().canvas.flush_events()
        train_accuracy = 100 * correct / total
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss:.4f}, Accuracy: {train_accuracy:.2f}%")
        ax[1].clear()
        epoch_loss_plot.append(train_accuracy)
        ax[1].plot(np.arange(len(epoch_loss_plot))+1,epoch_loss_plot,'--o')
        ax[1].set_xlabel("Epoch")
        ax[1].set_ylabel("Accuracy")
        ax[1].grid(True) 
        plt.draw()
        plt.pause(0.001)
        plt.gcf().canvas.flush_events()
    plt.show()

    # torch.save(model.state_dict(), 'trained_model.pth')


def train_model_with_parameters_update_visualization(model, train_loader, val_loader, num_epochs, criterion, optimizer, device):
    """
    Trains the model and visualizes parameter updates using TensorBoard.

    Args:
        model (torch.nn.Module): The PyTorch model to train.
        train_loader (DataLoader): DataLoader for training data.
        val_loader (DataLoader): DataLoader for validation data.
        num_epochs (int): Number of epochs for training.
        criterion (torch.nn.Module): Loss function.
        optimizer (torch.optim.Optimizer): Optimizer for training.
        device (torch.device): Device to run the model on.
    """
    writer = SummaryWriter()  # Initialize TensorBoard writer
    model.to(device)
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        
        for i, (radar_data, labels) in enumerate(train_loader):
            radar_data, labels = radar_data.to(device), labels.to(device)
            x1 = radar_data[:, 0:1, :, :]
            x2 = radar_data[:, 1:2, :, :]
            x3 = radar_data[:, 2:3, :, :]

            optimizer.zero_grad()
            outputs = model(x1, x2, x3)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

            # Log parameter updates to TensorBoard
            for name, param in model.named_parameters():
                writer.add_histogram(f'{name}_weights', param, epoch * len(train_loader) + i)
                if param.grad is not None:
                    writer.add_histogram(f'{name}_grads', param.grad, epoch * len(train_loader) + i)
        
        train_accuracy = 100 * train_correct / train_total
        writer.add_scalar("Loss/Train", train_loss, epoch)
        writer.add_scalar("Accuracy/Train", train_accuracy, epoch)

        # Validation phase
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for radar_data, labels in val_loader:
                radar_data, labels = radar_data.to(device), labels.to(device)
                x1 = radar_data[:, 0:1, :, :]
                x2 = radar_data[:, 1:2, :, :]
                x3 = radar_data[:, 2:3, :, :]
                outputs = model(x1, x2, x3)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_accuracy = 100 * val_correct / val_total
        writer.add_scalar("Loss/Validation", val_loss, epoch)
        writer.add_scalar("Accuracy/Validation", val_accuracy, epoch)
        
        print(f"Epoch {epoch + 1}/{num_epochs}, "
              f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%, "
              f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%")

    writer.close()

# Test Function with Confusion Matrix
def test_model(model, test_loader, device, idx_to_gesture):
    # model.eval()
    correct, total = 0, 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for radar_data, labels in test_loader:
            radar_data, labels = radar_data.to(device), labels.to(device)
            x1 = radar_data[:, 0:1, :, :]
            x2 = radar_data[:, 1:2, :, :]
            x3 = radar_data[:, 2:3, :, :]
            outputs = model(x1, x2, x3)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")

    # Compute confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=[idx_to_gesture[i] for i in range(len(idx_to_gesture))],
                yticklabels=[idx_to_gesture[i] for i in range(len(idx_to_gesture))])
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.show()

# Visualization Function
def visualize_sample(dataset, use_dataloader=True, sample_index=50):
    if use_dataloader:
        dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
        one_sample = next(iter(dataloader))
        batch, labels = one_sample
        input, label = batch[0, :, :, :], labels[0]
    else:
        input, label = dataset[sample_index]

    radar_data_matrix = input[0, :, :]
    plt.figure(figsize=(10, 6))
    plt.imshow(radar_data_matrix.numpy(), aspect='auto', cmap='viridis')
    plt.colorbar(label="Amplitude")
    plt.title(f"Sample Visualization - Label: {label}")
    plt.xlabel("Fast Time Bins")
    plt.ylabel("Slow Time Bins")
    plt.show()

def runradarmisoCNNapp():
    app = QApplication.instance()  # Check if an instance already exists
    if not app:  # If no instance exists, create one
        app = QApplication(sys.argv)
    app.setStyleSheet(ssp.config.appSTYLESHEET)
    
    window = RadarMisoCNNApp()
    window.show()
    app.exec_() 

class RadarMisoCNNApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Hand Gesture Classification Using Radar Signals and Deep Learning")
        self.default_folder = "/home/moein/Documents/MATLAB/Examples/R2024a/supportfiles/SPT/data/uwb-gestures"
        self.default_folder = "C:/Users/moein.ahmadi/OneDrive - University of Luxembourg/ThingsToDo/Max/HG Nature/data/uwb-gestures"
        self.initUI()

    def initUI(self):
        # Main layout
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout()
        central_widget.setLayout(main_layout)

        # Form layout for inputs
        form_layout = QFormLayout()
        self.data_folder_input = QLineEdit(self)
        self.data_folder_input.setText(self.default_folder)  # Set default folder
        self.data_folder_browse_button = QPushButton("Browse")
        self.data_folder_browse_button.clicked.connect(self.browse_folder)
        folder_layout = QHBoxLayout()
        folder_layout.addWidget(self.data_folder_input)
        folder_layout.addWidget(self.data_folder_browse_button)
        form_layout.addRow("Dataset Folder:", folder_layout)

        self.epochs_input = QSpinBox(self)
        self.epochs_input.setMinimum(1)
        self.epochs_input.setValue(3)
        form_layout.addRow("Epochs:", self.epochs_input)

        self.batch_size_input = QSpinBox(self)
        self.batch_size_input.setMinimum(1)
        self.batch_size_input.setValue(32)
        form_layout.addRow("Batch Size:", self.batch_size_input)

        self.learning_rate_input = QDoubleSpinBox(self)
        self.learning_rate_input.setMinimum(0.0001)
        self.learning_rate_input.setValue(0.5)
        self.learning_rate_input.setDecimals(4)
        form_layout.addRow("Learning Rate (x*1e-3):", self.learning_rate_input)

        self.split_train_input = QDoubleSpinBox(self)
        self.split_train_input.setMinimum(0.1)
        self.split_train_input.setMaximum(0.9)
        self.split_train_input.setSingleStep(0.05)
        self.split_train_input.setValue(0.7)
        form_layout.addRow("Train Split Ratio:", self.split_train_input)

        self.split_val_input = QDoubleSpinBox(self)
        self.split_val_input.setMinimum(0.05)
        self.split_val_input.setMaximum(0.9)
        self.split_val_input.setSingleStep(0.05)
        self.split_val_input.setValue(0.15)
        form_layout.addRow("Validation Split Ratio:", self.split_val_input)

        # Add "Clutter Removal" checkbox
        self.clutter_removal_checkbox = QCheckBox("Enable Clutter Removal", self)
        self.clutter_removal_checkbox.setChecked(True)
        form_layout.addRow(self.clutter_removal_checkbox)

        main_layout.addLayout(form_layout)

        # Buttons for actions in rows
        # Row 1
        row1_layout = QHBoxLayout()
        self.train_button = QPushButton("Train")
        self.train_button.clicked.connect(self.train_model)
        self.test_button = QPushButton("Test")
        self.test_button.clicked.connect(self.test_model)
        row1_layout.addWidget(self.train_button)
        row1_layout.addWidget(self.test_button)
        main_layout.addLayout(row1_layout)

        # Row 2
        row2_layout = QHBoxLayout()
        self.visualize_button = QPushButton("Visualize Network")
        self.visualize_button.clicked.connect(self.visualize_model)
        self.visualize_samples_button = QPushButton("Visualize Samples")
        self.visualize_samples_button.clicked.connect(self.visualize_samples)
        row2_layout.addWidget(self.visualize_button)
        row2_layout.addWidget(self.visualize_samples_button)
        main_layout.addLayout(row2_layout)

        # Row 3
        row3_layout = QHBoxLayout()
        self.visualize_image_button = QPushButton("Visualize Model as Image")
        self.visualize_image_button.clicked.connect(self.visualize_model_as_image)
        self.visualize_params_button = QPushButton("Visualize Parameter Updates (bash:tensorboard --logdir=runs)")
        self.visualize_params_button.clicked.connect(self.visualize_parameter_updates)
        row3_layout.addWidget(self.visualize_image_button)
        row3_layout.addWidget(self.visualize_params_button)
        main_layout.addLayout(row3_layout)

        # Status display
        self.status_label = QLabel("Status: Ready")
        main_layout.addWidget(self.status_label)

    def browse_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Dataset Folder", self.default_folder)
        if folder:
            self.data_folder_input.setText(folder)

    def train_model(self):
        try:
            data_folder = self.data_folder_input.text()
            if not os.path.isdir(data_folder):
                QMessageBox.critical(self, "Error", "Invalid dataset folder!")
                return

            dataset = RadarGestureDataset(data_folder,clutter_removal=self.clutter_removal_checkbox.isChecked())
            idx_to_gesture = {v: k for k, v in dataset.classes.items()}
            train_ratio = self.split_train_input.value()
            val_ratio = self.split_val_input.value()

            train_size = int(train_ratio * len(dataset))
            val_size = int(val_ratio * len(dataset))
            test_size = len(dataset) - train_size - val_size

            train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
            train_loader = DataLoader(train_dataset, batch_size=self.batch_size_input.value(), shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=self.batch_size_input.value(), shuffle=False)

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            num_classes = len(dataset.classes)
            model = MisoCNN(num_classes)
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=self.learning_rate_input.value()*1e-3)

            self.status_label.setText("Status: Training...")
            train_model(model, train_loader, val_loader, self.epochs_input.value(), criterion, optimizer, device)
            torch.save(model.state_dict(), os.path.join(ssp.config.temp_folder, "trained_model.pth"))
            self.status_label.setText("Status: Training completed.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"An error occurred during training: {str(e)}")

    def test_model(self):
        try:
            data_folder = self.data_folder_input.text()
            if not os.path.isdir(data_folder):
                QMessageBox.critical(self, "Error", "Invalid dataset folder!")
                return
            

            dataset = RadarGestureDataset(data_folder,clutter_removal=self.clutter_removal_checkbox.isChecked())
            idx_to_gesture = {v: k for k, v in dataset.classes.items()}
            train_ratio = self.split_train_input.value()
            val_ratio = self.split_val_input.value()

            train_size = int(train_ratio * len(dataset))
            val_size = int(val_ratio * len(dataset))
            test_size = len(dataset) - train_size - val_size

            _, _, test_dataset = random_split(dataset, [train_size, val_size, test_size])
            test_loader = DataLoader(test_dataset, batch_size=self.batch_size_input.value(), shuffle=False)

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            num_classes = len(dataset.classes)
            model = MisoCNN(num_classes)

            model_path, _ = QFileDialog.getOpenFileName(self, "Load Model", ssp.config.temp_folder, "PyTorch Models (*.pth)")
            if not model_path:
                QMessageBox.warning(self, "Warning", "No model file selected!")
                return
            QApplication.processEvents()
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.to(device)
            self.status_label.setText("Status: Testing...")
            test_model(model, test_loader, device, idx_to_gesture)
            self.status_label.setText("Status: Testing completed.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"An error occurred during testing: {str(e)}")
    def visualize_model(self):
        try:
            import io
            from contextlib import redirect_stdout
            from torchsummary import summary

            num_classes = 12  # Set this according to the dataset or modify as needed
            input_width,input_height = 90 , 189
            model = MisoCNN(num_classes=num_classes)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model.to(device)

            # Capture the output of the summary function
            buffer = io.StringIO()
            with redirect_stdout(buffer):
                summary(model, input_size=[(1, input_width, input_height), (1, input_width, input_height), (1, input_width, input_height)])
            model_summary = buffer.getvalue()

            # Create a scrollable dialog
            scrollable_dialog = QDialog(self)
            scrollable_dialog.setWindowTitle("Model Summary")
            scrollable_dialog.setMinimumSize(600, 400)

            layout = QVBoxLayout(scrollable_dialog)

            scroll_area = QScrollArea(scrollable_dialog)
            scroll_area.setWidgetResizable(True)
            layout.addWidget(scroll_area)

            inner_widget = QWidget()
            scroll_area.setWidget(inner_widget)
            inner_layout = QVBoxLayout(inner_widget)

            summary_label = QLabel(model_summary)
            summary_label.setTextInteractionFlags(Qt.TextSelectableByMouse)  # Allow text selection
            summary_label.setWordWrap(True)
            inner_layout.addWidget(summary_label)

            # Add close button
            close_button = QPushButton("Close", scrollable_dialog)
            close_button.clicked.connect(scrollable_dialog.accept)
            layout.addWidget(close_button)

            scrollable_dialog.exec_()

        except Exception as e:
            QMessageBox.critical(self, "Error", f"An error occurred during model visualization: {str(e)}")
    def visualize_samples(self):
        try:
            data_folder = self.data_folder_input.text()
            if not os.path.isdir(data_folder):
                QMessageBox.critical(self, "Error", "Invalid dataset folder!")
                return

            # Load the dataset
            dataset = RadarGestureDataset(data_folder, 1, clutter_removal=self.clutter_removal_checkbox.isChecked())

            # Get the total length of the dataset
            dataset_length = len(dataset)

            # Define the number of samples to plot in each batch
            batch_size = 10
            fig, axs = plt.subplots(2, 5, figsize=(15, 6))
            axs = axs.flatten()
                
            # Plot all samples in batches of 'batch_size'
            for batch_start in range(0, dataset_length, batch_size):
                # Determine the end of the current batch
                batch_end = min(batch_start + batch_size, dataset_length)
                
                # Create subplots for the current batch
                
                for i, ax in enumerate(axs):
                    # Check if we're out of dataset samples
                    if batch_start + i >= batch_end:
                        ax.axis("off")  # Turn off any unused subplot
                        continue
                    
                    idx = batch_start + i
                    radar_data, label = dataset[idx]
                    radar_image = radar_data[0].numpy()  # Select the first channel (left)
                    ax.clear()
                    ax.imshow(radar_image, aspect='auto', cmap='viridis')
                    ax.set_title(f"Label: {label.item()} : {radar_image.shape}")
                    ax.axis("off")

                plt.draw()
                plt.pause(0.001)
                plt.gcf().canvas.flush_events()
                plt.tight_layout()
                
            plt.show()

        except Exception as e:
            QMessageBox.critical(self, "Error", f"An error occurred while visualizing samples: {str(e)}")
    def visualize_model_as_image(self):
        try:
            os.environ["PATH"] += os.pathsep + "C:/Program Files/Graphviz/bin"
            from torchviz import make_dot

            # Create a dummy input to pass through the model
            num_classes = 12  # Adjust as needed
            model = MisoCNN(num_classes=num_classes)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model.to(device)
            dummy_input = [
                torch.randn(1, 1, 90, 189).to(device),
                torch.randn(1, 1, 90, 189).to(device),
                torch.randn(1, 1, 90, 189).to(device),
            ]

            # Generate model visualization
            graph = make_dot(model(*dummy_input), params=dict(model.named_parameters()))
            graph.format = "png"
            graph.render("model_visualization")

            # Display the image in a scrollable dialog
            dialog = QDialog(self)
            dialog.setWindowTitle("Model Visualization")
            dialog.setMinimumSize(800, 600)

            layout = QVBoxLayout(dialog)

            scroll_area = QScrollArea(dialog)
            scroll_area.setWidgetResizable(True)
            layout.addWidget(scroll_area)

            inner_widget = QWidget()
            scroll_area.setWidget(inner_widget)
            inner_layout = QVBoxLayout(inner_widget)

            label = QLabel()
            pixmap = QPixmap(os.path.join(ssp.config.temp_folder, "model_visualization.png"))
            label.setPixmap(pixmap)
            label.setScaledContents(True)
            inner_layout.addWidget(label)

            # Add close button
            close_button = QPushButton("Close", dialog)
            close_button.clicked.connect(dialog.accept)
            layout.addWidget(close_button)

            dialog.exec_()

        except Exception as e:
            QMessageBox.critical(self, "Error", f"An error occurred while visualizing the model: {str(e)}")
    def visualize_parameter_updates(self):
        try:
            data_folder = self.data_folder_input.text()
            if not os.path.isdir(data_folder):
                QMessageBox.critical(self, "Error", "Invalid dataset folder!")
                return

            dataset = RadarGestureDataset(data_folder,1,clutter_removal=self.clutter_removal_checkbox.isChecked())
            idx_to_gesture = {v: k for k, v in dataset.classes.items()}
            train_ratio = self.split_train_input.value()
            val_ratio = self.split_val_input.value()

            train_size = int(train_ratio * len(dataset))
            val_size = int(val_ratio * len(dataset))
            test_size = len(dataset) - train_size - val_size

            train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
            train_loader = DataLoader(train_dataset, batch_size=self.batch_size_input.value(), shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=self.batch_size_input.value(), shuffle=False)

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            num_classes = len(dataset.classes)
            model = MisoCNN(num_classes)
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=self.learning_rate_input.value()*1e-3)

            self.status_label.setText("Status: Training...")
            train_model_with_parameters_update_visualization(model, train_loader, val_loader, self.epochs_input.value(), criterion, optimizer, device)
            self.status_label.setText("Status: Training completed.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"An error occurred during training: {str(e)}")
