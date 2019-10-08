# ESN

This model is a replication of the Working Memory (WM) model using a Recurrent Neural Network (RNN) of the Echo State Network (ESN) type used by Razvan Pascanu and Herbert Jaeger in ["A Neurodynamical Model for Working Memory"](https://www.sciencedirect.com/science/article/pii/S0893608010001899) for [The ReScience Journal](http://rescience.github.io/).

This model is based from the article, and was built from scratch in Python3 to be as close as possible as the one described in the paper.

## Description of the model

This model is described in further details in [ReScience replication](LINK)).

## How to use it

The model has been split into multiple subfiles, for it to be modular and easily modifiable.
The first and most important file is *ESN.py*, which contains all the architecture of the model. When launched, it trains the working memory units and the output weights, and tests them, printing out their error rates and other feedbacks (see the ReScience article for further details about testing). It uses some given fonts with a random seed, which can be modifiable (and retested) if needed. The test is only done one time. For more instances, see *main.py*. The following command can be used to test the network:

```bash
    python3 ESN.py
```

If you want to modify the different variables defining the weights dimensions, caracteristics and/or training time of the network, feel free to modify them in the beginning of the file:

```python
    # Training times
    self.train_characters_Wmem = int(10000) # Only Wmem is computed -> 10000 characters sequence
    self.train_characters_Wout = int(49000) # Wout is computed      -> 49000 characters sequence
    self.test_characters = int(35000)       # Testing               -> 35000 characters sequence

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
    self.Wmem = np.empty((self.WM, (self.K + self.N + self.WM)))
    self.Wout = np.empty((self.L, (self.K + self.N)))
```

The second important file is
