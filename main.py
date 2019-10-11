from ESN import ESN
from alphascii import Alphascii
import numpy as np
import json
import sys
import os
import pickle

def save_object(obj, filename):
    with open(filename, 'wb') as output:  # Overwrites any existing file.
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)

print("\n------- 30 INSTANCES OF ESN WITH MAIN.PY -------\n")


# Number of instances to test
inst = 30

# Characteristics
K = 13 # 12 input units + bias
N = 1200 # Reservoir units -> 1200
L = 65 # Output units -> 65
WM = 6 # Feedback units -> 6

# Choose font with argument, freemono by default
if len(sys.argv) > 1:
    font = str(sys.argv[1])
else:
    fonts = "freemono"

# Directory name
dirname = "data/results/{}".format(font)
if not os.path.exists(dirname):
    os.mkdir(dirname)
    print("Directory", dirname, "created ")
print("Font: " + font)
print("Results will be saved in {}".format(dirname))

# Seed in argument, random by default
if len(sys.argv) > 2:
    seed = int(sys.argv[2])
else:
    import time
    seed = int((time.time()*10**6) % 4294967295)
try:
    np.random.seed(seed)
    print("\nSeed:", seed)
except:
    print("!!! WARNING !!!: ESN seed was not set correctly.")

# Counters for errors
bracket_errors = {"fn" : np.empty(inst), "fn_per_brackets" : np.empty(inst), "fn_per_char" : np.empty(inst), "fn_per_T" : np.empty(inst), "fp" : np.empty(inst), "fp_per_brackets" : np.empty(inst), "fp_per_char" : np.empty(inst), "fp_per_T" : np.empty(inst)}
counter = {
    "character": {"(" : np.empty(inst), ")" : np.empty(inst), "[" : np.empty(inst), "]" : np.empty(inst), "@" : np.empty(inst), "Other" : np.empty(inst)},
    "fp_increase": {"(" : np.empty(inst), ")" : np.empty(inst), "[" : np.empty(inst), "]" : np.empty(inst), "@" : np.empty(inst), "Other" : np.empty(inst)},
    "fp_decrease": {"(" : np.empty(inst), ")" : np.empty(inst), "[" : np.empty(inst), "]" : np.empty(inst), "@" : np.empty(inst), "Other" : np.empty(inst)}
}
Wmem = np.empty((inst, 6, K + N + WM)) # All Wmem values for averages
errors_y = np.empty(inst)
errors_y_alphabet = np.empty((inst, 65))

for i in range(inst):
    print("\n\nINSTANCE {}/{}\n".format(i+1, inst))

    esn = ESN(font, seed + 4*i)

    # Training Wmem
    esn.train_Wmem()

    # Training Wout
    esn.train_Wout()

    # Testing the ESN
    esn.test()

    bracket_errors["fn"][i] = esn.bracket_errors["fn"]
    bracket_errors["fn_per_brackets"][i] = esn.bracket_errors["fn_per_brackets"]
    bracket_errors["fn_per_char"][i] = esn.bracket_errors["fn_per_char"]
    bracket_errors["fn_per_T"][i] = esn.bracket_errors["fn_per_T"]
    bracket_errors["fp"][i] = esn.bracket_errors["fp"]
    bracket_errors["fp_per_brackets"][i] = esn.bracket_errors["fp_per_brackets"]
    bracket_errors["fp_per_char"][i] = esn.bracket_errors["fp_per_char"]
    bracket_errors["fp_per_T"][i] = esn.bracket_errors["fp_per_T"]

    counter["character"]["("][i] = esn.counter["character"]["("]
    counter["fp_increase"]["("][i] = esn.counter["fp_increase"]["("]
    counter["fp_decrease"]["("][i] = esn.counter["fp_decrease"]["("]
    counter["character"][")"][i] = esn.counter["character"][")"]
    counter["fp_increase"][")"][i] = esn.counter["fp_increase"][")"]
    counter["fp_decrease"][")"][i] = esn.counter["fp_decrease"][")"]
    counter["character"]["["][i] = esn.counter["character"]["["]
    counter["fp_increase"]["["][i] = esn.counter["fp_increase"]["["]
    counter["fp_decrease"]["["][i] = esn.counter["fp_decrease"]["["]
    counter["character"]["]"][i] = esn.counter["character"]["]"]
    counter["fp_increase"]["]"][i] = esn.counter["fp_increase"]["]"]
    counter["fp_decrease"]["]"][i] = esn.counter["fp_decrease"]["]"]
    counter["character"]["@"][i] = esn.counter["character"]["@"]
    counter["fp_increase"]["@"][i] = esn.counter["fp_increase"]["@"]
    counter["fp_decrease"]["@"][i] = esn.counter["fp_decrease"]["@"]
    counter["character"]["Other"][i] = esn.counter["character"]["Other"]
    counter["fp_increase"]["Other"][i] = esn.counter["fp_increase"]["Other"]
    counter["fp_decrease"]["Other"][i] = esn.counter["fp_decrease"]["Other"]

    Wmem[i] = esn.Wmem
    errors_y[i] = esn.errors_y
    errors_y_alphabet[i] = esn.errors_y_alphabet
    print("")


# Save values of bracket_errors, counter, errors_y, Wmem and Wout
save_object(bracket_errors, "{}/bracket_errors.pkl".format(dirname))
save_object(counter, "{}/counter.pkl".format(dirname))
save_object(Wmem, "{}/Wmem.pkl".format(dirname))
save_object(errors_y, "{}/errors_y.pkl".format(dirname))
save_object(errors_y_alphabet, "{}/errors_y_alphabet.pkl".format(dirname))
save_object(seed, "{}/seed.pkl".format(dirname))


# Print out all results
print("False negatives: {0:.1f} +/- {1:.1f}, {2:.2f} +/- {3:.2f}%, {4:.2f} +/- {5:.3f}%, {6:.3f} +/- {7:.3f}%,".format(np.mean(bracket_errors["fn"]), np.std(bracket_errors["fn"]), np.mean(bracket_errors["fn_per_brackets"]), np.std(bracket_errors["fn_per_brackets"]), np.mean(bracket_errors["fn_per_char"]), np.std(bracket_errors["fn_per_char"]), np.mean(bracket_errors["fn_per_T"]), np.std(bracket_errors["fn_per_T"])))
print("False positives: {0:.1f} +/- {1:.1f}, {2:.2f} +/- {3:.2f}%, {4:.2f} +/- {5:.3f}%, {6:.3f} +/- {7:.3f}%,".format(np.mean(bracket_errors["fp"]), np.std(bracket_errors["fp"]), np.mean(bracket_errors["fp_per_brackets"]), np.std(bracket_errors["fp_per_brackets"]), np.mean(bracket_errors["fp_per_char"]), np.std(bracket_errors["fp_per_char"]), np.mean(bracket_errors["fp_per_T"]), np.std(bracket_errors["fp_per_T"])))
print("Total: {0:.1f} +/- {1:.1f}, {2:.2f} +/- {3:.2f}%, {4:.2f} +/- {5:.3f}%, {6:.3f} +/- {7:.3f}%,".format(np.mean(bracket_errors["fn"] + bracket_errors["fp"]), np.std(bracket_errors["fn"] + bracket_errors["fp"]), np.mean(bracket_errors["fn_per_brackets"] + bracket_errors["fp_per_brackets"]), np.std(bracket_errors["fn_per_brackets"] + bracket_errors["fp_per_brackets"]), np.mean(bracket_errors["fn_per_char"] + bracket_errors["fp_per_char"]), np.std(bracket_errors["fn_per_char"] + bracket_errors["fp_per_char"]), np.mean(bracket_errors["fn_per_T"] + bracket_errors["fp_per_T"]), np.std(bracket_errors["fn_per_T"] + bracket_errors["fp_per_T"])))
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

alphabet = esn.train_alphascii_Wmem.alphabet
print("10 biggest error rate per character: ")
idx = np.argsort(np.mean(errors_y_alphabet, axis = 0))[::-1]
for i in idx:
    if i < 10:
        print("{}: {:.2%} +/- {:.2%}".format(alphabet[i], np.mean(errors_y_alphabet[:,i]), np.std(errors_y_alphabet[:,i])))

print("\nErrors output: {0:.2%} +/- {1:.2%}\n".format(np.mean(errors_y), np.std(errors_y)))
