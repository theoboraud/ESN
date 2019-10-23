"""
@author: Theophile BORAUD
t.boraud@warwick.co.uk
Copyright 2019, Theophile BORAUD, Anthony STROCK, All rights reserved.
"""

import json
import sys
import os
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from ESN import ESN
from alphascii import Alphascii
import matplotlib.cm as cm
import matplotlib.patches as mpatches

font = ""
if len(sys.argv) > 1:
    font = str(sys.argv[1])
else:
    font = "freemono"

dirname = "data/PCA/{}".format(font)
esnDirname = "data/test/{}".format(font)

files = ("{}/PC_X.npy".format(dirname), "{}/PC_U.npy".format(dirname), "{}/length_x.npy".format(dirname))

reload = False # Change to True to compute again

# Generate data only if not already generated. Delete every file in files list to be able to generate new data
if not np.all([os.path.exists(file) for file in files]) or reload == True:

    esn = ESN()

    esn.Win = np.load("{}/Win.npy".format(esnDirname))
    esn.W = np.load("{}/W.npy".format(esnDirname))
    esn.Wb = np.load("{}/Wb.npy".format(esnDirname))
    esn.Wmem = np.load("{}/Wmem.npy".format(esnDirname))
    esn.Wout = np.load("{}/Wout.npy".format(esnDirname))

    print("\nGenerating data...")
    length_x = np.empty(7, dtype = np.int)
    for i in range(7):
        print("\n------ Memory: {} ------".format(i))
        alphascii = Alphascii("PCA", 6500, seed = esn.seed + i, set_i = i)

        if i == 0:
            U, X = esn.test(alphascii)
            length_x[i] = int(X.shape[0])
        else:
            u, x = esn.test(alphascii)
            U = np.append(U, u, axis = 0)
            X = np.append(X, x, axis = 0)

            length_x[i] = int(x.shape[0])

    pca_X = PCA(n_components = 2)
    pca_X.fit(X)
    PC_X = pca_X.transform(X)

    pca_U = PCA(n_components = 1)
    pca_U.fit(U)
    PC_U = pca_U.transform(U)

    np.save("{}/PC_X".format(dirname), PC_X)
    np.save("{}/PC_U".format(dirname), PC_U)
    np.save("{}/length_x".format(dirname), length_x)

else:
    print("\nLoading data...\n")
    PC_X = np.load(files[0])
    PC_U = np.load(files[1])
    length_x = np.load(files[2])


def lighten_color(color, amount = 0.5):
    """
    Lightens the given color by multiplying (1-luminosity) by the given amount.
    Input can be matplotlib color string, hex string, or RGB tuple.

    Examples:
    >> lighten_color('g', 0.3)
    >> lighten_color('#F034A3', 0.6)
    >> lighten_color((.3,.55,.1), 0.5)
    """
    import matplotlib.colors as mc
    import colorsys
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])


colors = ['red', 'orange', 'yellow', 'green', 'turquoise', 'blue', 'purple']
fig = plt.figure()
ax = fig.add_subplot(111, projection = '3d')
idx = 0
patches = []
print("Generating figures...\n")
for i in range(7):
    index = np.random.permutation(np.arange(idx + 100, idx + length_x[i]))[:6000]
    index = np.arange(idx + 100, idx + length_x[i])
    ax.scatter(PC_X[index,0], PC_X[index,1], PC_U[index,0], s = 0.1, color = colors[i])
    ax.scatter(PC_X[index,0], PC_X[index,1], np.min(PC_U) - 0.5, s = 0.1, color = lighten_color(colors[i], 1.3))
    idx += length_x[i]
    patches.append(mpatches.Patch(color = colors[i], label = 'M = {:d}'.format(i)))

ax.set_xlabel('Reservoir PC 1')
ax.set_ylabel('Reservoir PC 2')
ax.set_zlabel('Input PC 1')
ax.legend(handles = patches)
plt.savefig(dirname + "/fig.png")
plt.draw()
plt.show(block = True)
