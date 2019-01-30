import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg
from ipywidgets import *
from IPython.display import *
import alphascii



# ------------------------------------------- #
# ---------------- FUNCTIONS ---------------- #
# ------------------------------------------- #


def set_seed(seed=None):
    """
    Create the seed (for random values) variable if None
    """

    if seed is None:
        import time
        seed = int((time.time()*10**6) % 4294967295)
    try:
        np.random.seed(seed)
        print("Seed:", seed)
    except:
        print("!!! WARNING !!!: Seed was not set correctly.")
    return seed


def plot_figure(data, data2 = None, title = ""):
    """
    Plot the figure of the given data using MatPlotLib

    Args:
        data (array of float): Dataset we want to plot
        title (string): String to print as the title
    """

    f = plt.figure(figsize = [20, 4])
    ax = f.add_subplot(1,1,1)
    ax.set_xlabel('Time')
    f.suptitle(title)
    ax.plot(data, color='green')
    if(not(data2 is None)):
        ax.plot(data2, color='blue')


def random_W(N):
    """
    Generate random weights for the reservoir

    Args:
        N (int): Number of reservoir units

    Returns:
        N x N matrix: Reservoir weights
    """

    W = np.zeros((N, N))
    # Generate random locations for the reservoir non-zero connections
    locations = [(i,j) for i in range(N) for j in range(N)]
    locations = np.random.permutation(locations)

    for i in range(nonzero_W):
        W[locations[i, 0], locations[i, 1]] = np.random.choice((-W_value, W_value))

    # Spectral value
    spectral_radius = 0.5
    radius = np.max(np.abs(np.linalg.eig(W)[0]))
    W *= spectral_radius / radius

    return W



def random_data(low, high, size):
    """
    Generate a random dataset for training and testing purposes

    Args:
        loc (float): mean
        scale (float): standard deviation
        size (int or tuple of ints):

    Returns:
        The dataset, in a form of a size x dim array
    """

    dataset = np.random.uniform(low, high, size)
    return dataset


def x_update(W, x_n, Win, u_n1, Wfb, y_n):
    """
    Compute the x_n+1 reservoir state

    Args:
        W (N x N matrix): reservoir weight matrix
        x_n (N vector): Previous reservoir state
        Win (N x K matrix): Input weight matrix
        u_n1 (K vector): Input signal
        Wfb (N x L matrix): Output feedback reservoir weight matrix
        y_n (L vector): Output feedback signal

    Returns:
        N-dimensional array of float: x_n+1
    """
    #print("Shapes:", W.shape, x_n.shape, Win.shape, u_n1.shape, Wfb.shape, y_n.shape)
    return np.tanh(np.dot(W, x_n) + np.dot(Win, u_n1) + np.dot(Wfb, y_n))


def get_Wout(X, Y):
    """
    Generate the ouput weights matrix

    Args:
        X (N x T matrix): Predictions matrix
        Y (K x T matrix): Target output

    Returns:
        K x N matrix: Output weights
    """

    #Wout = np.dot(Y, np.linalg.pinv(X))
    Wout = np.dot(Y, np.dot(X.T, np.linalg.inv(np.dot(X, X.T) + resWout * np.identity(X.shape[0]))))
    return Wout


def train(input, Y, T):
    """
    Trains the reservoir with training data for N times

    Args:
        input (size x K matrix):
        Y (L vector):
        T (int): Training time

    Returns:
        N x T matrix: Output weights matrix
    """

    # Compute a random x_n at first
    x_n = np.zeros(N)
    # Will store all values of x_n during time T
    X = np.empty((N, T))

    for i in range(1, T):

        # Compute the input u_n
        u_n = input[:,i]

        # Compute the target y_n
        y_n = Y[:,i-1]

        # Add x_n to X
        X[:,i-1] = x_n

        # Compute x_n+1
        x_n = x_update(W, x_n, Win, u_n, Wfb, y_n)

    Wout = get_Wout(X, Y)

    return Wout


def predict(input, Wout, Y, T):
    """
    Test the output weights on a new dataset

    Args:
        input (size x K matrix): Testing dataset
        Wout (K x N matrix): Output weights
        Y (L vector): Target output

    """

    costs = np.empty(T) # Store all error costs
    predictions = np.empty(T)
    targets = np.empty(T)

    # New x_n
    x_n = np.random.uniform(-1, 1, N)

    for i in range(T):

        u_n = input[:, i]

        # Compute the target y_n
        y_n = np.dot(Wout, x_n)

        # Compute the cost
        prediction = np.dot(Wout, x_n)
        target = Y[:, i-1]

        costs[i] = np.square(np.linalg.norm(prediction - target))

        predictions[i] = prediction

        targets[i] = target

        # Compute x_n+1
        x_n = x_update(W, x_n, Win, u_n, Wfb, y_n)

    plot_figure(targets, predictions, "Target outputs and ESN predictions")
    plot_figure(costs, "Errors")
    plt.show()

    cost = np.sum(costs)
    cost = np.sqrt(cost / T)


    print("Error: {:.1e}".format(cost))


# ------------------------------------------- #
# ---------------- VARIABLES ---------------- #
# ------------------------------------------- #


# Number of trainings and tests
train_time = 10000
test_time = 35000

# Number of reservoir units
N = 1200

# Reservoir non-zeros connections
nonzero_W = 12000

# Reservoir weights value
W_value = 0.1540

# Number of inputs units
K = 13

# Number of outputs units
L = 65

# Number of WM units
WM = 6

# Generate the weight restraint in Wout
resWout = 1e-4

# Generate the reservoir weights matrix
W = random_W(N)


# Generate the input weights matrix
Win = np.random.choice((0, -0.5, 0.5), (N, K), True, (0.8, 0.1, 0.1))

# Generate the output feedback reservoir weights matrix
Wfb = np.random.choice((-0.4, 0.4), (N, WM))



# ------------------------------------------- #
# ----------------- TESTING ----------------- #
# ------------------------------------------- #


# Initiate the seed value
seed = set_seed()

# Create training dataset
train_alphascii = Alphascii(train_time)
data_train = train_alphascii.image
Y_train = train_alphascii.sequence


# Create testing dataset
data_test = MackeyGlass[None, train_time:-1]
#data_test = random_data(-1, 1, (K, test_time))
#data_test = np.sin(np.arange(test_time)/100)[None, :]
Y_test = MackeyGlass[None, train_time+1:] # Target values for testing
test_time -=1

# Learning step
Wout = train(data_train, Y_train, train_time)

# Predictive step
predict(data_test, Wout, Y_test, test_time)
