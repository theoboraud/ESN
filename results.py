"""
@author: Theophile BORAUD
t.boraud@warwick.co.uk
Copyright 2019, Theophile BORAUD, Anthony STROCK, All rights reserved.
"""

from PIL import Image
import sys
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pickle
from alphascii import Alphascii
import warnings
warnings.filterwarnings('ignore', '.*output shape of zoom.*')

# Select font in argument, freemono by default
if len(sys.argv) > 1:
    font = str(sys.argv[1])
else:
    font = "freemono"

dirname = "data/results/{}".format(font)
bracket_errors = pickle.load(open('{}/bracket_errors.pkl'.format(dirname), 'rb'))
counter = pickle.load(open("{}/counter.pkl".format(dirname), 'rb'))
Wmem = pickle.load(open("{}/Wmem.pkl".format(dirname), 'rb'))
errors_y = pickle.load(open("{}/errors_y.pkl".format(dirname), 'rb'))
errors_y_alphabet = pickle.load(open("{}/errors_y_alphabet.pkl".format(dirname), 'rb'))
seed = pickle.load(open("{}/seed.pkl".format(dirname), 'rb'))

# Print out all results
print("\nResults from directory {}".format(dirname))
print("")
print("Seed:", seed)
print("")
print("False negatives: {0:.1f} +/- {1:.1f}, {2:.2%} +/- {3:.2%}, {4:.2%} +/- {5:.3%}, {6:.3%} +/- {7:.3%},".format(np.mean(bracket_errors["fn"]), np.std(bracket_errors["fn"]), np.mean(bracket_errors["fn_per_brackets"]), np.std(bracket_errors["fn_per_brackets"]), np.mean(bracket_errors["fn_per_char"]), np.std(bracket_errors["fn_per_char"]), np.mean(bracket_errors["fn_per_T"]), np.std(bracket_errors["fn_per_T"])))
print("False positives: {0:.1f} +/- {1:.1f}, {2:.2%} +/- {3:.2%}, {4:.2%} +/- {5:.3%}, {6:.3%} +/- {7:.3%},".format(np.mean(bracket_errors["fp"]), np.std(bracket_errors["fp"]), np.mean(bracket_errors["fp_per_brackets"]), np.std(bracket_errors["fp_per_brackets"]), np.mean(bracket_errors["fp_per_char"]), np.std(bracket_errors["fp_per_char"]), np.mean(bracket_errors["fp_per_T"]), np.std(bracket_errors["fp_per_T"])))
print("Total: {0:.1f} +/- {1:.1f}, {2:.2%} +/- {3:.2%}, {4:.2%} +/- {5:.3%}, {6:.3%} +/- {7:.3%},".format(np.mean(bracket_errors["fn"] + bracket_errors["fp"]), np.std(bracket_errors["fn"] + bracket_errors["fp"]), np.mean(bracket_errors["fn_per_brackets"] + bracket_errors["fp_per_brackets"]), np.std(bracket_errors["fn_per_brackets"] + bracket_errors["fp_per_brackets"]), np.mean(bracket_errors["fn_per_char"] + bracket_errors["fp_per_char"]), np.std(bracket_errors["fn_per_char"] + bracket_errors["fp_per_char"]), np.mean(bracket_errors["fn_per_T"] + bracket_errors["fp_per_T"]), np.std(bracket_errors["fn_per_T"] + bracket_errors["fp_per_T"])))
print("")

print("\"(\" ({0:.1f} +/- {1:.1f}): increases {2:.2f} +/- {3:.2f}, decreases {4:.1f} +/- {5:.1f}".format(np.mean(counter["character"]["("]), np.std(counter["character"]["("]), np.mean(counter["fp_increase"]["("]), np.std(counter["fp_increase"]["("]), np.mean(counter["fp_decrease"]["("]), np.std(counter["fp_decrease"]["("])))
print("\")\" ({0:.1f} +/- {1:.1f}): increases {2:.2f} +/- {3:.2f}, decreases {4:.1f} +/- {5:.1f}".format(np.mean(counter["character"][")"]), np.std(counter["character"][")"]), np.mean(counter["fp_increase"][")"]), np.std(counter["fp_increase"][")"]), np.mean(counter["fp_decrease"][")"]), np.std(counter["fp_decrease"][")"])))
print("\"[\" ({0:.1f} +/- {1:.1f}): increases {2:.2f} +/- {3:.2f}, decreases {4:.1f} +/- {5:.1f}".format(np.mean(counter["character"]["["]), np.std(counter["character"]["["]), np.mean(counter["fp_increase"]["["]), np.std(counter["fp_increase"]["["]), np.mean(counter["fp_decrease"]["["]), np.std(counter["fp_decrease"]["["])))
print("\"]\" ({0:.1f} +/- {1:.1f}): increases {2:.2f} +/- {3:.2f}, decreases {4:.1f} +/- {5:.1f}".format(np.mean(counter["character"]["]"]), np.std(counter["character"]["]"]), np.mean(counter["fp_increase"]["]"]), np.std(counter["fp_increase"]["]"]), np.mean(counter["fp_decrease"]["]"]), np.std(counter["fp_decrease"]["]"])))
print("\"@\" ({0:.1f} +/- {1:.1f}): increases {2:.2f} +/- {3:.2f}, decreases {4:.1f} +/- {5:.1f}".format(np.mean(counter["character"]["@"]), np.std(counter["character"]["@"]), np.mean(counter["fp_increase"]["@"]), np.std(counter["fp_increase"]["@"]), np.mean(counter["fp_decrease"]["@"]), np.std(counter["fp_decrease"]["@"])))
print("\"Other\" ({0:.1f} +/- {1:.1f}): increases {2:.2f} +/- {3:.2f}, decreases {4:.1f} +/- {5:.1f}".format(np.mean(counter["character"]["Other"]), np.std(counter["character"]["Other"]), np.mean(counter["fp_increase"]["Other"]), np.std(counter["fp_increase"]["Other"]), np.mean(counter["fp_decrease"]["Other"]), np.std(counter["fp_decrease"]["Other"])))
print("")

print("Input to WM-units average weights: {0:.4f} +/- {1:.4f}".format(np.mean(np.abs(Wmem[:, :, :14])), np.std(np.abs(Wmem[:, :, :14]))))
print("Reservoir to WM-units average weights: {0:.4f} +/- {1:.4f}".format(np.mean(np.abs(Wmem[:, :, 14:1214])), np.std(np.abs(Wmem[:, :, 14:1214]))))
print("WM-units to WM-units average weights: {0:.4f} +/- {1:.4f}".format(np.mean(np.abs(Wmem[:, :, 1214:])), np.std(np.abs(Wmem[:, :, 1214:]))))
print("")

alphabet = Alphascii("Training", 1, seed = 0).alphabet
print("10 biggest error rate per character: ")
idx = np.argsort(np.mean(errors_y_alphabet, axis = 0))[::-1]
for i in idx:
    if i < 10:
        print("{}: {:.2%} +/- {:.2%}".format(alphabet[i], np.mean(errors_y_alphabet[:,i]), np.std(errors_y_alphabet[:,i])))

print("\nErrors output: {0:.2%} +/- {1:.2%}\n".format(np.mean(errors_y), np.std(errors_y)))
