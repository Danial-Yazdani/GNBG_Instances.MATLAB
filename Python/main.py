import os
import numpy as np
from scipy.io import loadmat
import random

# Get the current script's directory
current_dir = os.path.dirname(os.path.abspath(__file__))

# Define the path to the folder where you want to read/write files
folder_path = os.path.join(current_dir)


# Initialization
RunNumber = 5
AcceptancePoints = np.full(RunNumber, np.nan)
Error = np.full(RunNumber, np.nan)
ProblemIndex = 22  # Choose a problem instance range from f1 to f24

# Check if the chosen problem index is within the valid range
if ProblemIndex < 1 or ProblemIndex > 24:
    raise ValueError("Invalid problem index. Please choose a number from 1 to 24.")




for RunCounter in range(1, RunNumber + 1):
    print(f'RunCounter={RunCounter}')
    # Preparation and loading of the GNBG parameters based on the chosen problem instance
    if 1 <= ProblemIndex <= 24:
        filename = f'f{ProblemIndex}.mat'
        GNBG = loadmat(os.path.join(folder_path, filename))['GNBG']
    else:
        raise ValueError('ProblemIndex must be between 1 and 24.')
    # Set a random seed for the optimizer
    np.random.seed()  # This uses a system-based source to seed the random number generator


x=GNBG['RotationMatrix']