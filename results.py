"""
@author: Theophile BORAUD
t.boraud@warwick.co.uk
Copyright 2019, Theophile BORAUD, Anthony STROCK, All rights reserved.
"""

from PIL import Image
import json
import sys
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

if len(sys.argv) > 1:
    dirname = str(sys.argv[1])
else:
    dirname = "data/Classic_Italic_Bold_BoldItalic"

bracket_errors = json.loads(open("{}/bracket_errors.json".format(dirname)).read())
counter = json.loads(open("{}/counter.json".format(dirname)).read())
errors_y = json.loads(open("{}/errors_outputs.json".format(dirname)).read())
U = np.load("{}/U.npy".format(dirname))
M = np.load("{}/M.npy".format(dirname))
X = np.load("{}/X.npy".format(dirname))
Win = np.load("{}/Win.npy".format(dirname))
W = np.load("{}/W.npy".format(dirname))
Wb = np.load("{}/Wb.npy".format(dirname))
Wmem = np.load("{}/Wmem.npy".format(dirname))
Wout = np.load("{}/Wout.npy".format(dirname))
img = Image.open("{}/testing.png".format(dirname))

print("")
print("Results for font(s) FreeMono{}".format(dirname[5:]))
print("")
print("Bracket false negative: {} (Curly brackets: {:.2%}) (Characters: {:.2%}) (Time steps: {:.2%})".format(bracket_errors["fn"], bracket_errors["fn_per_brackets"], bracket_errors["fn_per_char"], bracket_errors["fn_per_T"]))
print("Bracket false positive: {} (Curly brackets: {:.2%}) (Characters: {:.2%}) (Time steps: {:.2%})".format(bracket_errors["fp"], bracket_errors["fp_per_brackets"], bracket_errors["fp_per_char"], bracket_errors["fp_per_T"]))
print("")
print("\"(\" ({} times in sequence): increased {} times and decreased {} times".format(counter["character"]["("], counter["fp_increase"]["("], counter["fp_decrease"]["("]))
print("\")\" ({} times in sequence): increased {} times and decreased {} times".format(counter["character"][")"], counter["fp_increase"][")"], counter["fp_decrease"][")"]))
print("\"[\" ({} times in sequence): increased {} times and decreased {} times".format(counter["character"]["["], counter["fp_increase"]["["], counter["fp_decrease"]["["]))
print("\"]\" ({} times in sequence): increased {} times and decreased {} times".format(counter["character"]["]"], counter["fp_increase"]["]"], counter["fp_decrease"]["]"]))
print("\"@\" ({} times in sequence): increased {} times and decreased {} times".format(counter["character"]["@"], counter["fp_increase"]["@"], counter["fp_decrease"]["@"]))
print("Other character ({} times in sequence): increased {} times and decreased {} times".format(counter["character"]["Other"], counter["fp_increase"]["Other"], counter["fp_decrease"]["Other"]))
print("")
print("Output error rate: {:.2%}".format(errors_y))
print("")
print("Average input to WM-units weights: {}".format(np.mean(np.abs(Wmem[:, :13]))))
print("Average reservoir to WM-units weights: {}".format(np.mean(np.abs(Wmem[:, 13:1213]))))
print("Average WM-units to WM-units weights: {}".format(np.mean(np.abs(Wmem[:, 1213:]))))
print("")
img.show()
