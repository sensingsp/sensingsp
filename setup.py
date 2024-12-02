from setuptools import setup, find_packages
from pathlib import Path

# Read the long description from the README.md file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding='utf-8')

# Define the required dependencies
install_requires = [
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
]

# Define the setup
setup(
    name='sensingsp',
    version='1.1.0',
    packages=find_packages(),
    install_requires=install_requires,
    url='https://sensingsp.github.io/',
    license='MIT',
    author='Moein Ahmadi',
    author_email='moein.ahmadi@uni.lu, gmoein@gmail.com',
    description=(
        'SensingSP™: Open-source library for simulating electromagnetic sensing systems '
        'and implementing signal processing and machine learning algorithms. Developed by '
        'the Signal Processing Applications in Radar and Communications (SPARC) group at '
        'the University of Luxembourg with support from IEE S.A.'
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
