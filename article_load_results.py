from PIL import Image
import sys
import json
import warnings
import os
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.patches as mpatches
import pickle
from tqdm import tqdm


os.system('python3 results.py freemono')
os.system('python3 results.py inconsolata')
os.system('python3 PCA.py')
