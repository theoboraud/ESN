from ESN import ESN
from alphascii import Alphascii
import numpy as np

inst = 30

dirname = "data/Test"
bracket_errors = {"fn" : np.empty(inst), "fn_per_brackets" : np.empty(inst), "fn_per_char" : np.empty(inst), "fn_per_T" : np.empty(inst), "fp" : np.empty(inst), "fp_per_brackets" : np.empty(inst), "fp_per_char" : np.empty(inst), "fp_per_T" : np.empty(inst)}
counter = {
    "character": {"(" : np.empty(inst), ")" : np.empty(inst), "[" : np.empty(inst), "]" : np.empty(inst), "@" : np.empty(inst), "Other" : np.empty(inst)},
    "fp_increase": {"(" : np.empty(inst), ")" : np.empty(inst), "[" : np.empty(inst), "]" : np.empty(inst), "@" : np.empty(inst), "Other" : np.empty(inst)},
    "fp_decrease": {"(" : np.empty(inst), ")" : np.empty(inst), "[" : np.empty(inst), "]" : np.empty(inst), "@" : np.empty(inst), "Other" : np.empty(inst)}
}

Wmem = np.empty((inst, 6, 1219))
errors_y = np.empty(inst)
errors_y_alphabet = np.empty((inst, 65))

esn = ESN()
alphabet = np.empty(65)
seed = 1000

for i in range(inst):
    print("\nINSTANCE {}/{}\n".format(i+1, inst) )

    # Create training dataset for Wmem -> 1st training stage
    train_alphascii_Wmem = Alphascii("Training", esn.train_characters_Wmem, seed)
    if i == 0:
        alphabet = train_alphascii_Wmem.alphabet
    # Training Wmem
    esn.train_Wmem(train_alphascii_Wmem)

    # Create training dataset for Wout -> 2nd training stage
    train_alphascii_Wout = Alphascii("Training", esn.train_characters_Wout, seed + 1)
    # Training Wout
    esn.train_Wout(train_alphascii_Wout)

    # Create testing dataset -> testing stage
    test_alphascii = Alphascii("Testing", esn.test_characters, seed + 2)

    # Increase seed for randomized input generation of next instance
    seed += 3

    # Testing the ESN
    esn.test(test_alphascii)

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

print("False negatives: {0:.3f} +/- {1:.3f}, {2:.3f} +/- {3:.3f}%, {4:.3f} +/- {5:.3f}%, {6:.3f} +/- {7:.3f}%,".format(np.mean(bracket_errors["fn"]), np.std(bracket_errors["fn"]), np.mean(bracket_errors["fn_per_brackets"]), np.std(bracket_errors["fn_per_brackets"]), np.mean(bracket_errors["fn_per_char"]), np.std(bracket_errors["fn_per_char"]), np.mean(bracket_errors["fn_per_T"]), np.std(bracket_errors["fn_per_T"])))
print("False positives: {0:.3f} +/- {1:.3f}, {2:.3f} +/- {3:.3f}%, {4:.3f} +/- {5:.3f}%, {6:.3f} +/- {7:.3f}%,".format(np.mean(bracket_errors["fp"]), np.std(bracket_errors["fp"]), np.mean(bracket_errors["fp_per_brackets"]), np.std(bracket_errors["fp_per_brackets"]), np.mean(bracket_errors["fp_per_char"]), np.std(bracket_errors["fp_per_char"]), np.mean(bracket_errors["fp_per_T"]), np.std(bracket_errors["fp_per_T"])))
print("Total: {0:.3f} +/- {1:.3f}, {2:.3f} +/- {3:.3f}%, {4:.3f} +/- {5:.3f}%, {6:.3f} +/- {7:.3f}%,".format(np.mean(bracket_errors["fn"] + bracket_errors["fp"]), np.std(bracket_errors["fn"] + bracket_errors["fp"]), np.mean(bracket_errors["fn_per_brackets"] + bracket_errors["fp_per_brackets"]), np.std(bracket_errors["fn_per_brackets"] + bracket_errors["fp_per_brackets"]), np.mean(bracket_errors["fn_per_char"] + bracket_errors["fp_per_char"]), np.std(bracket_errors["fn_per_char"] + bracket_errors["fp_per_char"]), np.mean(bracket_errors["fn_per_T"] + bracket_errors["fp_per_T"]), np.std(bracket_errors["fn_per_T"] + bracket_errors["fp_per_T"])))
print("")

print("\"(\" ({0:.3f} +/- {1:.3f}): increases {2:.3f} +/- {3:.3f}, decreases {4:.3f} +/- {5:.3f}".format(np.mean(counter["character"]["("]), np.std(counter["character"]["("]), np.mean(counter["fp_increase"]["("]), np.std(counter["fp_increase"]["("]), np.mean(counter["fp_decrease"]["("]), np.std(counter["fp_decrease"]["("])))
print("\")\" ({0:.3f} +/- {1:.3f}): increases {2:.3f} +/- {3:.3f}, decreases {4:.3f} +/- {5:.3f}".format(np.mean(counter["character"][")"]), np.std(counter["character"][")"]), np.mean(counter["fp_increase"][")"]), np.std(counter["fp_increase"][")"]), np.mean(counter["fp_decrease"][")"]), np.std(counter["fp_decrease"][")"])))
print("\"[\" ({0:.3f} +/- {1:.3f}): increases {2:.3f} +/- {3:.3f}, decreases {4:.3f} +/- {5:.3f}".format(np.mean(counter["character"]["["]), np.std(counter["character"]["["]), np.mean(counter["fp_increase"]["["]), np.std(counter["fp_increase"]["["]), np.mean(counter["fp_decrease"]["["]), np.std(counter["fp_decrease"]["["])))
print("\"]\" ({0:.3f} +/- {1:.3f}): increases {2:.3f} +/- {3:.3f}, decreases {4:.3f} +/- {5:.3f}".format(np.mean(counter["character"]["]"]), np.std(counter["character"]["]"]), np.mean(counter["fp_increase"]["]"]), np.std(counter["fp_increase"]["]"]), np.mean(counter["fp_decrease"]["]"]), np.std(counter["fp_decrease"]["]"])))
print("\"@\" ({0:.3f} +/- {1:.3f}): increases {2:.3f} +/- {3:.3f}, decreases {4:.3f} +/- {5:.3f}".format(np.mean(counter["character"]["@"]), np.std(counter["character"]["@"]), np.mean(counter["fp_increase"]["@"]), np.std(counter["fp_increase"]["@"]), np.mean(counter["fp_decrease"]["@"]), np.std(counter["fp_decrease"]["@"])))
print("\"Other\" ({0:.3f} +/- {1:.3f}): increases {2:.3f} +/- {3:.3f}, decreases {4:.3f} +/- {5:.3f}".format(np.mean(counter["character"]["Other"]), np.std(counter["character"]["Other"]), np.mean(counter["fp_increase"]["Other"]), np.std(counter["fp_increase"]["Other"]), np.mean(counter["fp_decrease"]["Other"]), np.std(counter["fp_decrease"]["Other"])))
print("")

print("Input to WM-units average weights: {0:.4f} +/- {1:.4f}".format(np.mean(np.abs(Wmem[:, :, :14])), np.std(np.abs(Wmem[:, :, :14]))))
print("Reservoir to WM-units average weights: {0:.4f} +/- {1:.4f}".format(np.mean(np.abs(Wmem[:, :, 14:1214])), np.std(np.abs(Wmem[:, :, 14:1214]))))
print("WM-units to WM-units average weights: {0:.4f} +/- {1:.4f}".format(np.mean(np.abs(Wmem[:, :, 1214:])), np.std(np.abs(Wmem[:, :, 1214:]))))
print("")

print("10 biggest error rate per character: ")
idx = np.argsort(np.mean(errors_y_alphabet, axis = 0))[::-1]
for i in idx:
    if i < 10:
        print("{}: {:.2%} +/- {:.2%}".format(alphabet[i], np.mean(errors_y_alphabet[:,i]), np.std(errors_y_alphabet[:,i])))

print("Errors output: {0:.2%} +/- {1:.2%}".format(np.mean(errors_y), np.std(errors_y)))

print("errors_y: {}".format(errors_y))
