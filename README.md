# Replicating the results of Dynamics-Aware Unsupervised Discovery of Skills
Reproducing the results of the Dynamics-Aware Unsupervised Discovery of Skills paper (ICLR 2020)

## Requirements
A dockerfile is present for use on systems with AMD GPUs - such as the RX Vega 56 used by one developer. Using this is entirely optional. The requirements are: 

* PyTorch
* Open AI Gym >= 0.23.1
* MuJoCo >= 2.1.0
* Python (Miniconda/Anaconda/Miniforge is strongly suggested)

## MacOS M1 Installation Guidelines
To install MuJoCo for M1 macOS systems, please view the following tutorial (Minforge has to be used as the Python environment instead of Anaconda/Miniconda).

https://github.com/openai/mujoco-py/issues/682

Ensure that you can import mujoco_py, gym and torch  packages before attempting to run the code.

## Running on colab
Copy and paste all files on the main branch into cells on collab, deleting the import lines of packages starting with "."
Install MuJoCo using the following instuctions: https://colab.research.google.com/drive/1KGMZdRq6AemfcNscKjgpRzXqfhUtCf-V?usp=sharing
