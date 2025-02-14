import numpy as np
import cv2
from numba import cuda
import os
import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QPushButton, QCheckBox, QSpinBox, QLabel, QHBoxLayout,QComboBox
import sensingsp as ssp

import random


def read_cells(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # Remove comments and empty lines
    pattern_lines = [line.strip() for line in lines if line.strip() and not line.startswith('!')]

    # Determine the dimensions of the pattern
    height = len(pattern_lines)
    width = max(len(line) for line in pattern_lines)

    # Initialize the NumPy array with zeros (dead cells)
    pattern_array = np.zeros((height, width), dtype=int)

    # Populate the array with live cells
    for i, line in enumerate(pattern_lines):
        for j, char in enumerate(line):
            if char == 'O':
                pattern_array[i, j] = 1

    return pattern_array


# CUDA kernel to update the Game of Life grid
@cuda.jit
def game_of_life_kernel(current_grid, next_grid):
    x, y = cuda.grid(2)  # Current cell coordinates
    rows, cols = current_grid.shape

    if x < rows and y < cols:  # Ensure within bounds
        # Count live neighbors
        live_neighbors = 0
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                nx, ny = (x + dx) % rows, (y + dy) % cols  # Wrap around for periodic boundaries
                live_neighbors += current_grid[nx, ny]

        # Apply Conway's Game of Life rules
        if current_grid[x, y] == 1:
            next_grid[x, y] = 1 if live_neighbors in [2, 3] else 0
        else:
            next_grid[x, y] = 1 if live_neighbors == 3 else 0

# CUDA kernel to generate the frame
@cuda.jit
def generate_frame_kernel(grid, frame, cell_size, live_color, dead_color):
    x, y = cuda.grid(2)
    rows, cols = grid.shape
    frame_height, frame_width = rows * cell_size, cols * cell_size

    if x < frame_height and y < frame_width:  # Ensure within bounds
        # Determine which cell this pixel belongs to
        cell_row = x // cell_size
        cell_col = y // cell_size

        # Determine the color based on cell state
        if grid[cell_row, cell_col] == 1:
            frame[x, y, 0] = live_color[0]  # B
            frame[x, y, 1] = live_color[1]  # G
            frame[x, y, 2] = live_color[2]  # R
        else:
            frame[x, y, 0] = dead_color[0]
            frame[x, y, 1] = dead_color[1]
            frame[x, y, 2] = dead_color[2]

def save_game_of_life_video_cuda(grid=[],grid_size=(100, 100), steps=1000, fps=30, cell_size=10, video_file="game_of_life_cuda.mp4"):
    rows, cols = grid_size
    frame_size = (rows * cell_size, cols * cell_size, 3)

    # Initialize the grid
    # grid = np.random.choice([0, 1], size=grid_size, p=[0.8, 0.2]).astype(np.int32)
    d_current_grid = cuda.to_device(grid)
    d_next_grid = cuda.to_device(np.zeros_like(grid))

    # Allocate GPU memory for the frame
    d_frame = cuda.to_device(np.zeros(frame_size, dtype=np.uint8))

    # Video writer setup
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_file, fourcc, fps, (cols * cell_size, rows * cell_size))

    # Colors
    live_color = (0, 200, 0)  # Green for live cells
    dead_color = (0, 0, 0)    # Black for dead cells

    # CUDA grid/block dimensions
    threads_per_block = (16, 16)
    blocks_per_grid_x = (rows + threads_per_block[0] - 1) // threads_per_block[0]
    blocks_per_grid_y = (cols + threads_per_block[1] - 1) // threads_per_block[1]
    blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)

    frame_blocks_x = (frame_size[0] + threads_per_block[0] - 1) // threads_per_block[0]
    frame_blocks_y = (frame_size[1] + threads_per_block[1] - 1) // threads_per_block[1]
    frame_blocks = (frame_blocks_x, frame_blocks_y)

    for step in range(steps):
        # Update the Game of Life grid
        game_of_life_kernel[blocks_per_grid, threads_per_block](d_current_grid, d_next_grid)
        d_current_grid, d_next_grid = d_next_grid, d_current_grid

        # Generate the frame
        generate_frame_kernel[frame_blocks, threads_per_block](
            d_current_grid, d_frame, cell_size, live_color, dead_color
        )

        # Copy frame back to host
        frame = d_frame.copy_to_host()

        # Write the frame to the video
        out.write(frame)

        # Optional progress log
        if step % 100 == 0:
            print(f"Step {step}/{steps} completed.")

    # Release the video writer
    out.release()
    print(f"Video saved as '{video_file}'")

def save_game_of_life_video_cpu(grid =[],grid_size=(100, 100), steps=1000, fps=30, cell_size=10, video_file="game_of_life_cpu.mp4"):
    rows, cols = grid_size
    frame_size = (rows * cell_size, cols * cell_size, 3)

    # Initialize the grid
    # grid = np.random.choice([0, 1], size=grid_size, p=[0.8, 0.2]).astype(np.int32)

    # Video writer setup
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_file, fourcc, fps, (cols * cell_size, rows * cell_size))

    # Colors
    live_color = (0, 200, 0)  # Green for live cells
    dead_color = (0, 0, 0)    # Black for dead cells
    steps0=int(steps/100)
    for step in range(steps):
        # Update the grid
        next_grid = np.zeros_like(grid)
        for x in range(rows):
            for y in range(cols):
                # Count live neighbors
                live_neighbors = sum(
                    grid[(x + dx) % rows, (y + dy) % cols]
                    for dx in [-1, 0, 1]
                    for dy in [-1, 0, 1]
                    if not (dx == 0 and dy == 0)
                )
                # Apply Conway's Game of Life rules
                if grid[x, y] == 1:
                    next_grid[x, y] = 1 if live_neighbors in [2, 3] else 0
                else:
                    next_grid[x, y] = 1 if live_neighbors == 3 else 0
        grid = next_grid

        # Generate the frame
        frame = np.zeros(frame_size, dtype=np.uint8)
        for x in range(rows):
            for y in range(cols):
                color = live_color if grid[x, y] == 1 else dead_color
                frame[x * cell_size:(x + 1) * cell_size, y * cell_size:(y + 1) * cell_size] = color

        # Write the frame to the video
        out.write(frame)

        # Optional progress log
        if step % steps0 == 0:
            
            # QApplication.processEvents()

            print(f"Step {step}/{steps} completed.")

    # Release the video writer
    out.release()
    print(f"Video saved as '{video_file}'")


class GameofLifeApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Game of Life Simulation")
        self.initUI()

    def initUI(self):
        central_widget = QWidget()
        layout = QVBoxLayout()

        # Checkbox for CUDA availability
        self.cuda_checkbox = QCheckBox("CUDA Available: " + str(ssp.config.CUDA_is_available))
        self.cuda_checkbox.setChecked(True)
        layout.addWidget(self.cuda_checkbox)

        directory_path = os.path.join(ssp.config.temp_folder,"Conways")
        os.makedirs(directory_path, exist_ok=True)
        N= len(os.listdir(directory_path))
        if N<100:
            ssp.utils.hub.download_zipfile_extract_remove("https://conwaylife.com/patterns/","all.zip",directory_path)
        sorted_file_names=[]
        if os.path.isdir(directory_path):
            files_with_sizes = [
                (file, os.path.getsize(os.path.join(directory_path, file)))
                for file in os.listdir(directory_path)
                if file.endswith('.cells')
            ]

            # Sort files by size
            sorted_files = sorted(files_with_sizes, key=lambda x: x[1])

            # Extract only the filenames, sorted by size
            sorted_file_names = [f"{file} : {size/1024} KB" for file, size in sorted_files]
            # 
            # self.pattern_combo.setCurrentIndex(0)
        # ComboBox for pattern selection
        pattern_layout = QHBoxLayout()
        pattern_layout.addWidget(QLabel("Pattern:"))
        self.pattern_combo = QComboBox()
        # self.pattern_combo.addItems([
        #     "Random", 
        #     "Block", 
        #     "Blinker", 
        #     "Glider", 
        #     "Toad", 
        #     "Beacon", 
        #     "Pulsar", 
        #     "Gosper Glider Gun", 
        #     "Lightweight Spaceship (LWSS)", 
        #     "Diehard"
        # ])      
        self.pattern_combo.addItems(sorted_file_names)  
        pattern_layout.addWidget(self.pattern_combo)
        layout.addLayout(pattern_layout)

        # Spin boxes for inputs
        grid_size_layout = QHBoxLayout()
        grid_size_layout.addWidget(QLabel("Grid Size:"))
        self.grid_rows_spinbox = QSpinBox()
        self.grid_rows_spinbox.setRange(10, 10000)
        self.grid_rows_spinbox.setValue(1080)
        if not ssp.config.CUDA_is_available:
            self.grid_rows_spinbox.setValue(108)
        grid_size_layout.addWidget(QLabel("Rows"))
        grid_size_layout.addWidget(self.grid_rows_spinbox)
        self.grid_cols_spinbox = QSpinBox()
        self.grid_cols_spinbox.setRange(10, 10000)
        self.grid_cols_spinbox.setValue(1920)
        if not ssp.config.CUDA_is_available:
            self.grid_cols_spinbox.setValue(100)
        grid_size_layout.addWidget(QLabel("Cols"))
        grid_size_layout.addWidget(self.grid_cols_spinbox)
        layout.addLayout(grid_size_layout)

        steps_layout = QHBoxLayout()
        steps_layout.addWidget(QLabel("Steps:"))
        self.steps_spinbox = QSpinBox()
        self.steps_spinbox.setRange(10, 10000)
        self.steps_spinbox.setValue(1000)
        
        if not ssp.config.CUDA_is_available:
            self.steps_spinbox.setValue(100)
        steps_layout.addWidget(self.steps_spinbox)
        layout.addLayout(steps_layout)

        fps_layout = QHBoxLayout()
        fps_layout.addWidget(QLabel("FPS:"))
        self.fps_spinbox = QSpinBox()
        self.fps_spinbox.setRange(1, 120)
        self.fps_spinbox.setValue(60)
        fps_layout.addWidget(self.fps_spinbox)
        layout.addLayout(fps_layout)

        cell_size_layout = QHBoxLayout()
        cell_size_layout.addWidget(QLabel("Cell Size:"))
        self.cell_size_spinbox = QSpinBox()
        self.cell_size_spinbox.setRange(1, 50)
        self.cell_size_spinbox.setValue(2)
        if not ssp.config.CUDA_is_available:
            self.cell_size_spinbox.setValue(10)
        cell_size_layout.addWidget(self.cell_size_spinbox)
        layout.addLayout(cell_size_layout)

        # Button to save video
        save_button = QPushButton("Save Game of Life Video")
        save_button.clicked.connect(self.save_video)
        layout.addWidget(save_button)

        # Button to open temp folder
        open_folder_button = QPushButton("Open Temp Folder")
        open_folder_button.clicked.connect(ssp.utils.open_temp_folder)
        layout.addWidget(open_folder_button)

        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)

    def save_video(self):
        f=self.pattern_combo.currentText().split(" : ")[0].split(".")[0]
        video_file = os.path.join(ssp.config.temp_folder, f"game_of_life_{f}.mp4")
        grid_size = (self.grid_rows_spinbox.value(), self.grid_cols_spinbox.value())
        steps = self.steps_spinbox.value()
        fps = self.fps_spinbox.value()
        cell_size = self.cell_size_spinbox.value()

        # Get selected pattern
        selected_pattern = self.pattern_combo.currentText()

        # Generate the initial grid based on the selected pattern
        grid = self.initialize_grid(grid_size, selected_pattern)

        if self.cuda_checkbox.isChecked() and ssp.config.CUDA_is_available:
            save_game_of_life_video_cuda(grid=grid,
                grid_size=grid_size, steps=steps, fps=fps, cell_size=cell_size, video_file=video_file
            )
        else:
            save_game_of_life_video_cpu(grid=grid,
                grid_size=grid_size, steps=steps, fps=fps, cell_size=cell_size, video_file=video_file
            )

    def initialize_grid(self, grid_size, pattern_name):
        """Initialize the grid based on the selected pattern."""
        grid = np.random.choice([0, 1], size=grid_size, p=[0.8, 0.2]).astype(np.int32)
        
        # Define the directory path
        directory_path = os.path.join(ssp.config.temp_folder,"Conways")
        thefile=self.pattern_combo.currentText().split(" : ")[0]
        # Step 1: Check if the directory exists
        if os.path.isdir(directory_path):
            # Step 2: List all files in the directory
            pattern = read_cells(os.path.join(directory_path , thefile))
            grid = np.zeros(grid_size, dtype=np.int32)
            grid = self.load_pattern(grid, pattern)
        return grid


    def load_pattern(self, grid, pattern):
        """Place a pattern at the center of the grid."""
        rows, cols = grid.shape
        pattern_rows, pattern_cols = pattern.shape
        start_x = (rows - pattern_rows) // 2
        start_y = (cols - pattern_cols) // 2
        grid[start_x:start_x+pattern_rows, start_y:start_y+pattern_cols] = pattern
        return grid

def runapp():
    app = QApplication.instance()  # Check if an instance already exists
    if not app:  # If no instance exists, create one
        app = QApplication(sys.argv)
    app.setStyleSheet(ssp.config.appSTYLESHEET)  # Set the stylesheet

    window = GameofLifeApp()
    window.show()
    app.exec_()
