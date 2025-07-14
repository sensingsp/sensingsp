from setuptools import setup, find_packages
from pathlib import Path

# Read the long description from the README.md file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding='utf-8')

# Define the required dependencies
install_requires = [
    'numpy<2.0.0',                # Ensure compatibility with modules compiled for NumPy 1.x
    'bpy>=4.0.0',                 # Blender Python API
    'matplotlib>=3.5.0',          # Plotting library
    'opencv-python>=4.5.0',       # OpenCV for computer vision
    'PyQt5>=5.15.0',              # PyQt5 for GUI applications
    'scipy>=1.9.0',               # Scientific computing
    'numba>=0.56.0',              # JIT compiler for Python
    'torch>=1.12.0',              # PyTorch for deep learning
    'scikit-learn>=1.0.0',        # Machine learning library
    'seaborn>=0.11.0',            # Statistical data visualization
    'tensorboard>=2.9.0',         # TensorBoard for visualizations
    'torchviz>=0.0.2',            # Visualization of PyTorch models
    'torchsummary>=1.5.0',        # Summary of PyTorch models
    'PyWavelets>=1.7.0',
    'pyserial>=3.5',
]

# Define the setup
setup(
    name='sensingsp',
    version='1.5.8',
    packages=find_packages(),
    install_requires=install_requires,
    url='https://sensingsp.github.io/',
    license='MIT',
    author='Moein Ahmadi',
    author_email='moein.ahmadi@uni.lu, gmoein@gmail.com',
    description=(
        'SensingSP™ is an open-source library designed for simulating ' 
        'electromagnetic sensing systems and implementing signal processing ' 
        'and machine learning algorithms.'
    ),
    long_description=long_description,
    long_description_content_type='text/markdown',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Developers',
        'Intended Audience :: Information Technology',
        'Topic :: Scientific/Engineering :: Information Analysis',
        'Topic :: Scientific/Engineering :: Physics',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.10',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.10',
    extras_require={
        'no_deps': [],  # Allows installing without any dependencies
    },
)
