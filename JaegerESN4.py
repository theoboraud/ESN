
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy import linalg
from ipywidgets import *
from IPython.display import *
from alphascii import Alphascii
from PIL import Image
import sys
import warnings
warnings.filterwarnings('ignore', '.*output shape of zoom.*')


# ------------------------------------------- #
# ------------------ CLASS ------------------ #
# ------------------------------------------- #


class JaegerESN:


# ------------------------------------------- #
# ---------------- FUNCTIONS ---------------- #
# ------------------------------------------- #


    def __init__(self):
        """
        Constructor of JaegerESN class

        Args:
            K (int): Number of input given by the alpascii objects (used to vary size of sequence data characters)

        Returns:
            JaegerESN object
        """
        # Set seed for whole network
        self.seed = self.set_seed()

        # Number of training and tests characters
        if len(sys.argv) > 1: # If argument is specified for testing purposes, multiply training and testing time by it
            mult = float(sys.argv[1])
        else:
            mult = 1
        self.train_characters_Wmem = int(10000 * mult) # Only Wmem is computed -> 10000 characters sequence
        self.train_characters_Wout = int(49000 * mult) # Wout is computed      -> 49000 characters sequence
        self.test_characters = int(35000 * mult) # Testing                      -> 35000 characters sequence

        # Input
        self.K = 13 # 12 input units + bias
        print("K =", self.K)
        self.U_bias = -0.5 # Input bias value

        # Reservoir
        self.N = 1200 # Reservoir units -> 1200
        print("N =", self.N)

        # Output
        self.L = 65 # Output units -> 65
        print("L =", self.L)
        self.WM = 6 # Feedback units -> 6
        print("WM =", self.WM)

        # Weights
        self.Win = np.random.choice((0, -0.5, 0.5), (self.N, self.K), True, (0.8, 0.1, 0.1)) # Input weight matrix, 80% zeros
        self.nonzero_W = 12000 # Reservoir non-zeros connections
        self.W_value = 0.1540 # Reservoir weights value (-0.1540 or +0.1540)
        self.W = self.random_W(self.N) # Reservoir weight matrix of size N x N
        self.Wb = np.random.choice((-0.4, 0.4), (self.N, self.WM)) # Feedback weight matrix
        #self.Wb = np.zeros((self.N, self.WM))
        self.res_Wout = 1e-4 # Weight restraint in Wout
        self.Wout_threshold = 0.5
        print("")


    def set_seed(self):
        """
        Create the seed (for random values) variable if none has been declared in sys.argv

        Returns:
            int: Seed
        """

        if len(sys.argv) > 2:
            seed = int(sys.argv[2])
        else:
            import time
            seed = int((time.time()*10**6) % 4294967295)
        try:
            np.random.seed(seed)
            print("Seed JaegerESN:", seed)
        except:
            print("!!! WARNING !!!: JaegerESN seed was not set correctly.")
        return seed


    def random_W(self, N):
        """
        Generate random weights connections for the reservoir

        Args:
            N (int): Number of reservoir units

        Returns:
            N x N matrix: Reservoir weights
        """

        W = np.zeros((N, N))
        # Generate random locations for the reservoir non-zero connections
        locations = [(i,j) for i in range(N) for j in range(N)]
        locations = np.random.permutation(locations)

        for i in range(self.nonzero_W):
            W[locations[i, 0], locations[i, 1]] = np.random.choice((-self.W_value, self.W_value))

        # Spectral value
        spectral_radius = 0.5
        radius = np.max(np.abs(np.linalg.eig(W)[0]))
        W *= spectral_radius / radius

        return W


    def add_bias(self, data, bias):
        """
        Add the bias at the end of the given dataset

        Args:
            data (X x Y matrix): Given dataset at the end of which the bias will be added
            bias (int): Bias to add

        Returns:
            X x Y+1 matrix: Bias-added dataset
        """

        new_data = np.empty((data.shape[0], data.shape[1] + 1))
        new_data[:,:-1] = data
        new_data[:,-1] = bias

        return new_data


    def train_Wmem(self, alphascii):
        """
        First training stage, where the reservoir-to-WM-units weights (Wmem) are computed

        Args:
            alphascii (object): Training Alphascii object

        Returns:
            WM x (K + N + WM) matrix: Wmem
        """

        U = jaeger.add_bias(alphascii.data, self.U_bias) # Inputs (T x K matrix)
        M = alphascii.nested_lvl_outputs # Target of WM-units (T x WM matrix)
        T = len(U) # Training time (int)

        # Compute a random x_n at first
        x_n = np.zeros(self.N)

        # Store all values of x_n during time T
        X = np.empty((T, self.N))
        X[0] = x_n
        m_n = M[0]

        print("TRAINING Wmem")

        for i in tqdm(range(1, T-1)):
            # Current u(n+1)
            u_n1 = U[i]
            # Update x(n) with x(n+1)
            x_n = self.x_n1(self.Win, u_n1, self.W, x_n, self.Wb, m_n)
            # Store x(n+1) into X
            X[i] = x_n
            # Update m_n
            m_n = M[i]

        Wmem = self.compute_Wmem(U, X, M)
        print("")
        """ ------UNCOMMENT TO PRINT TARGET AND PREDICTED WM-UNITS------
        M_test = np.empty((T, self.WM))
        X_test = np.empty((T, self.N))
        M_test[0] = M[0]
        X_test[0] = X[0]
        for i in tqdm(range(1, T)):
            X_test[i] = self.x_n1(self.Win, U[i], self.W, X_test[i-1], self.Wb, M_test[i-1])
            M_test[i] = self.m_n1(Wmem, U[i], X_test[i], M_test[i-1])
            #print("M({}) = {}".format(i, M[i]))
            #print("M_test({}) = {}".format(i, M_test[i]))

        img1 = self.concatenate_imgs(alphascii.sequence_img, self.img_WM_units(M))
        img = self.concatenate_imgs(img1, self.img_WM_units(M_test))
        img.show()
        img.save("data/train_Wmem.png")
        """
        return Wmem


    def compute_Wmem(self, U, X, M):
        """
        Equation for computing reservoir to WM-units weights (Wmem)

        Args:
            U (T x K matrix): Inputs activations during time T
            X (T x N matrix): Reservoir states during time T
            M (T x WM matrix): Feedback outputs activations during time T

        Returns:
            WM x (K + N + WM) matrix: Wmem
        """

        H = np.concatenate([U[1:], X[1:], M[:-1]], axis = 1)
        Wmem = np.dot(np.linalg.pinv(H), M[1:]).T

        print("Wmem has been computed")

        return Wmem


    def train_Wout(self, alphascii, Wmem):
        """
        Second training stage, where reservoir to output weights (Wout) are computed

        Args:
            alphascii (object): Training Alphascii object
            Wmem (WM x (K + N + WM)): Reservoir to WM-units weights

        Returns:
            L x (K + N) matrix: Wout
        """

        U = self.add_bias(alphascii.data, self.U_bias)# Inputs activations during time T (T x K matrix)
        M = alphascii.nested_lvl_outputs # Target of WM-units (T x WM matrix)
        Y = alphascii.sequence_outputs # Target output activations during time T (T x L matrix)
        T = len(U) # Training time (int)

        # Compute a random x_n at first
        x_n = np.random.rand(self.N)

        # Store all values of x_n during time T
        X = np.empty((T, self.N))
        X[0] = x_n

        print("TRAINING Wout")

        for i in tqdm(range(1, T)):
            # Current u(n+1)
            u_n1 = U[i]

            # Update x(n) with x(n+1)
            x_n = self.x_n1(self.Win, u_n1, self.W, x_n, self.Wb, M[i-1])

            # Store x(n+1) into X
            X[i] = x_n

        Wout = self.compute_Wout(U, X, Y)
        print("")

        """ ------UNCOMMENT TO PRINT TARGET AND PREDICTED OUTPUT UNITS------
        Y_test = np.empty((T, self.L))
        X_test = np.empty((T, self.N))
        Y_test[0] = Y[0]
        X_test[0] = X[0]
        for i in tqdm(range(1, T)):
            X_test[i] = self.x_n1(self.Win, U[i], self.W, X_test[i-1], self.Wb, M[i-1])
            Y_test[i] = np.dot(Wout, np.concatenate([U[i], X_test[i]]))
            #print("M({}) = {}".format(i, M[i]))
            #print("M_test({}) = {}".format(i, M_test[i]))

        img1 = self.concatenate_imgs(alphascii.image, self.img_WM_units(M))
        img2 = self.concatenate_imgs(self.img_outputs(Y), self.img_outputs(Y_test, ignore = np.where(np.isnan(Y[:,0]))[0]))
        img3 = self.concatenate_imgs(img2, self.img_outputs(Y_test, ignore = np.where(np.isnan(Y[:,0]))[0], set_max = True))
        img = self.concatenate_imgs(img1, img3)
        img.show()
        img.save("data/train_Wout_without_Wb.png")
        """

        return Wout


    def compute_Wout(self, U, X, Y):
        """
        Equation for computing reservoir to output weights (Wout)

        Args:
            U (T x K matrix): Inputs activations during time T
            X (T x N matrix): Reservoir states during time T
            Y (T x L matrix): Target output activations during time T

        Returns:
            L x (K + N) matrix: Wout
        """

        idx = np.where(np.isfinite(Y[:,0]))
        G = np.concatenate([U[idx], X[idx]], axis = 1)
        Wout = np.dot(np.linalg.pinv(G), Y[idx]).T

        print("Wout has been computed")

        return Wout


    def test(self, alphascii, Wmem, Wout):
        """
        Testing part

        Args:
            alphascii (object): Testing Alphascii object
            Wmem (WM x (K + N + WM)): Reservoir to WM-units weights
            Wout (L x (K + N)): Reservoir to output weight
        """

        U = self.add_bias(alphascii.data, self.U_bias)# Inputs activations during time T (T x K matrix)
        M = alphascii.nested_lvl_outputs # Target WM-units during time T (T x WM)
        Y = alphascii.sequence_outputs # Target outputs during time T (T x L)
        T = len(U) # Training time (int)

        # Compute initial x and m
        x_n = np.zeros(self.N)
        m_n = np.ones(self.WM) * -0.5

        # Store all values of predicted m and y
        predictions_m = np.empty((T, self.WM))
        predictions_y = np.empty((T, self.L))

        # Errror counters
        bracket_fn = 0 # Number of false negative errors
        bracket_fp = 0 # Number of false positive errors
        errors_m = np.zeros(T) # Store a 1 at index i if there is an error at i
        errors_y = 0 # Number of output errors

        print("TESTING")
        # TO CHANGE ------------
        cur_width = 0
        cur_char = 0
        char_ended = False

        # Characters counter
        characters_counter = {"(": 0, ")": 0, "[": 0, "]": 0, "@": 0, "Other": 0}
        fp_increase = {"(": 0, ")": 0, "[": 0, "]": 0, "@": 0, "Other": 0}
        fp_decrease = {"(": 0, ")": 0, "[": 0, "]": 0, "@": 0, "Other": 0}

        for i in tqdm(range(1, T)):
            # TO CHANGE ------------
            char_ended = False
            cur_width += 1
            if cur_width == alphascii.width_chars[cur_char]:
                char_ended = True
                cur_width = 0
                cur_char += 1
                if i < len(M):
                    last_M = M[i]
                self.increase_counters(characters_counter, alphascii.sequence_pxl[i])

            # Update all states
            u_n = U[i]
            x_n = self.x_n1(self.Win, u_n, self.W, x_n, self.Wb, m_n)
            y_n = np.dot(Wout, np.concatenate([u_n, x_n]))
            m_n = self.m_n1(Wmem, u_n, x_n, m_n)

            # False positives and negatives
            if (not np.array_equal(m_n, M[i])):
                if alphascii.sequence_pxl[i] != "{" and alphascii.sequence_pxl[i] != "}":
                    bracket_fp += 1
                    if sum(m_n) > sum(M[i]):
                        self.increase_counters(fp_increase, alphascii.sequence_pxl[i])
                    if sum(m_n) < sum(M[i]):
                        self.increase_counters(fp_decrease, alphascii.sequence_pxl[i])
                    m_n = M[i]
                    errors_m[i] = 1
                else:
                    if char_ended:
                        bracket_fn += 1
                        m_n = M[i]
                        errors_m[i] = 1

            predictions_m[i] = m_n # We store the value of m

            if np.argmax(y_n) != np.argmax(Y[i]) and not(np.any(np.isnan(Y[i]))): #
                errors_y += 1

            predictions_y[i] = y_n # We store the value of y

        print("")
        print("GENERATING FIGURES")
        print("")

        img1 = self.concatenate_imgs(alphascii.image, self.img_WM_units(M))
        img2 = self.concatenate_imgs(self.img_WM_units(predictions_m, errors_m), self.img_outputs(Y))
        img3 = self.concatenate_imgs(img2, self.img_outputs(predictions_y, ignore = np.where(np.isnan(Y[:,0]))[0]))
        img4 = self.concatenate_imgs(img3, self.img_outputs(predictions_y, ignore = np.where(np.isnan(Y[:,0]))[0], set_max = True))
        img = self.concatenate_imgs(img1, img4)
        img.show()
        img.save("data/test.png")

        errors_y /= alphascii.n_characters # Percentage error of y

        # Print results
        print("Bracket false negative: {} (Curly brackets: {:.2%}) (Characters: {:.2%}) (Time steps: {:.2%})".format(bracket_fn, bracket_fn/alphascii.n_brackets, bracket_fn/alphascii.n_characters, bracket_fn/T))
        print("Bracket false positive: {} (Curly brackets: {:.2%}) (Characters: {:.2%}) (Time steps: {:.2%})".format(bracket_fp, bracket_fp/alphascii.n_brackets, bracket_fp/alphascii.n_characters, bracket_fp/T))
        print("\"(\" ({} times in sequence): increased {} times and decreased {} times".format(characters_counter["("], fp_increase["("], fp_decrease["("]))
        print("\")\" ({} times in sequence): increased {} times and decreased {} times".format(characters_counter[")"], fp_increase[")"], fp_decrease[")"]))
        print("\"[\" ({} times in sequence): increased {} times and decreased {} times".format(characters_counter["["], fp_increase["["], fp_decrease["["]))
        print("\"]\" ({} times in sequence): increased {} times and decreased {} times".format(characters_counter["]"], fp_increase["]"], fp_decrease["]"]))
        print("\"@\" ({} times in sequence): increased {} times and decreased {} times".format(characters_counter["@"], fp_increase["@"], fp_decrease["@"]))
        print("Other character ({} times in sequence): increased {} times and decreased {} times".format(characters_counter["Other"], fp_increase["Other"], fp_decrease["Other"]))
        print("")
        print("Output error rate: {:.2%}".format(errors_y))
        print("")
        print("Fonts: {}".format(alphascii.fontfiles))


    def x_n1(self, Win, u_n1, W, x_n, Wb, m_n):
        """
        Compute the x(n+1) reservoir state with the following equation:
        x(n+1) = f(Win u(n+1) + W x(n) + Wb m(n))

        Args:
            W (N x N matrix): Reservoir weight matrix
            x_n (N vector): Previous reservoir state
            Win (N x K matrix): Input weight matrix
            u_n1 (K vector): Input signal
            Wb (N x L matrix): Output feedback reservoir weight matrix
            m_n (WM vector): Output feedback signal

        Returns:
            N vector: x(n+1)
        """

        return np.tanh(np.dot(Win, u_n1) + np.dot(W, x_n) + np.dot(Wb, m_n))


    def m_n1(self, Wmem, u_n1, x_n1, m_n):
        """
        Compute the m(n+1) WM-units activations state with the following equation:
        m(n+1) = fm(Wmem (u(n+1), x(n+1), m(n)))

        Args:
            Wmem (T x (K + N + WM) matrix): Reservoir to WM-units weights matrix
            u_n1 (K vector): Actual input activations
            x_n1 (N vector): Actual reservoir activations
            m_n (WM vector): Previous feedback output activations

        Returns:
            WM vector: m(n+1)
        """

        return self.fm(np.dot(Wmem, np.concatenate([u_n1, x_n1, m_n], axis = 0)))


    def fm(self, x):
        """
        Sharp threshold function

        Args:
            x: Variable to sharp

        Returns:
            Sharped variable
        """

        x = np.piecewise(x, [x <= 0, x > 0], [-0.5, 0.5])
        return x


    def increase_counters(self, counter, c):
        """
        Increase the counters of characters, false positive increase errors and false decrease errors

        Args:
            counter (dictionary): Dictionary containing the counters to increase
            c (string): Character for which we want to increase the counter
        """

        if c in counter.keys():
            counter[c] += 1
        elif c != "{" and c != "}":
            counter["Other"] += 1



    def img_WM_units(self, WM_units, errors = None):
        """
        Create the image representation of WM-units during time

        Args:
            WM_units (T x WM matrix): WM-units during time
            errors (T vector): Indicates if there is an error on time t (0 for no, 1 for yes)

        Returns:
            Image object: Image representation of the given WM_units
        """

        x = min(int(WM_units.shape[0]), 20000)
        y = WM_units.shape[1]
        img = Image.new('L', (x, 2*y + 1)) # Create an image for WM-units
        pixels = img.load() # Create the pixel map

        if errors is None: # If no errors, create an array of 0s
            errors = np.zeros(x)

        for i in range(x):    # For every col:
            for j in range(y):    # For every row:
                if WM_units[i, j] == 0.5:
                    if errors[i] == 1: # If there is an error on time i, print out the WM_units in gray
                        pixels[i, 2*j+1] = 120
                    else: # If no errors on time i, print out the WM-units in white
                        pixels[i, 2*j+1] = 255
        return img


    def img_outputs(self, Y, ignore = [], set_max = False):
        """
        Create the image representation of outputs activations during time

        Args:
            Y (T x L matrix): Outputs activations during time

        Returns:
            Image object: Image representation of the given outputs
        """

        x = min(int(Y.shape[0]), 20000)
        y = Y.shape[1]
        img = Image.new('L', (x, y)) # Create an image for outputs units
        pixels = img.load() # Create the pixel map

        for i in range(x):    # For every col:
            if np.any(np.isnan(Y[i])) or i in ignore:
                for j in range(y):
                    pixels[i,j] = 120
            #elif i in ignore:
                #i = int(i)
                #for j in range(y):
                    #pixels[i,j] = 120
            else:
                if set_max:
                    max_pxl = np.argmax(Y[i])
                    Y[i] = 0
                    Y[i, max_pxl] = 1
                for j in range(y):
                    pixels[i,j] = int(Y[i,j] * 255)

        return img


    def concatenate_imgs(self, img_1, img_2):
        """
        Concatenate vertically two images

        Args:
            img_1 (Image object): First of the two images to concatenate
            img_2 (Image object): Second of the two images to concatenate

        Returns:
            Image object: Concatenated image
        """

        (x, y1) = img_1.size
        (x, y2) = img_2.size
        y_inter = 5
        img = Image.new('L', (x, y1 + y2 + y_inter))
        img_inter = Image.new('L', (x, y_inter), 50)
        img.paste(img_1, (0, 0))
        img.paste(img_inter, (0, y1))
        img.paste(img_2, (0, y1 + y_inter))

        return img


# ------------------------------------------- #
# ----------------- TESTING ----------------- #
# ------------------------------------------- #

if __name__ == "__main__":

    jaeger = JaegerESN()

    # Create training dataset for Wmem -> 1st training stage
    train_alphascii_Wmem = Alphascii("Training", jaeger.train_characters_Wmem, jaeger.seed)
    # Training Wmem
    Wmem = jaeger.train_Wmem(train_alphascii_Wmem)

    # Create training dataset for Wout -> 2nd training stage
    train_alphascii_Wout = Alphascii("Training", jaeger.train_characters_Wout, jaeger.seed + 1)
    # Training Wout
    Wout = jaeger.train_Wout(train_alphascii_Wout, Wmem)

    # Create testing dataset -> testing stage
    test_alphascii = Alphascii("Testing", jaeger.test_characters, jaeger.seed + 2)
    # Testing the JaegerESN
    jaeger.test(test_alphascii, Wmem, Wout)
