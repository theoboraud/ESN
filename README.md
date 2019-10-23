
# ESN

This model is a replication of the Working Memory (WM) model using a Recurrent Neural Network (RNN) of the Echo State Network (ESN) type used by Razvan Pascanu and Herbert Jaeger in ["A Neurodynamical Model for Working Memory"](https://www.sciencedirect.com/science/article/pii/S0893608010001899) for [The ReScience Journal](http://rescience.github.io/).

This model is based from the article, and was built from scratch in Python3 to be as close as possible as the one described in the paper.

## Description of the model

This model is described in further details in the [ReScience replication article](LINK).

## How to use it

The model has been split into multiple subfiles, for it to be modular and easily modifiable.
The first and most important file is *ESN.py*, which contains all the architecture of the model. When launched, it trains the working memory units and the output weights, and tests them, printing out their error rates and other feedbacks (see the ReScience article for further details about testing). It uses some given fonts with a random seed, which can be modifiable (and retested) if needed. The test is only done one time. For more instances, see *main.py*. The following command can be used to test the network with one instance:

```bash
    python ESN.py
```

If you want to modify the different variables defining the weights dimensions, characteristics and/or training time of the network, feel free to modify them in the beginning of the file:

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

The second important file is *alphascii.py*, which contains the input generation method, using an alphabet of ASCII symbols. It generates a random generated sequence of characters picked from a dataset of symbol called *self.alphabet*, separated sometimes with curly brackets, determined by some rules.

```python
    self.alphabet = "abcdefghijklmnopqrstuvwxyz0123456789 !\"#$%&\'()*+,.-_/:;<=>?@€|[]§" # The sequence is build using random characters from the alphabet
```

During training, 70% of the time, the character will be a symbol from the alphabet, while there is a 15% chance of getting an opening curly bracket, and 15% chance of getting a closing curly bracket.
During testing, symbols are picked up 94% of the time, with an equal probability of 3% for each curly brackets. (There is also a submode for generating sequences called "PCA" where there is no curly brackets, used in *PCA.py*).
Whenever a curly bracket is opened, the bracket level increases by one, up to 6. Whenever a curly bracket is closed, this brackets level decreases by one, with a minimum of 0.
When a symbol is picked up, its actual value is also randomly chosen. 80% of the time, for a given character *i* and a bracket level *j*, the next character will be *i* + *j* + 1 modulo 65 (size of *self.alphabet*).
The other 20% of the time, the character will be randomly chosen among the 64 other possible characters.

Finally, the sequence is converted to an image, using *convert_sequence_to_img()*, creating a sub-image matrix for each character, with a random font from the given *fontfiles* at the beginning of the file. Every character image has a size of 12 and a width of 7, and is then reshaped into an image with a width of 6, 7 or 8 (randomly), with finally a salt-and-pepper Gaussian noise of amplitude 0.1 applied on all of them.

You can easily test this class by using the following command:

```bash
    python alphascii.py
```

In order to obtain similar results as in the former article, we need to initialize and test the network 30 times in a row, and then display the results. To do so, we use *main.py*, which will create 30 instances of *ESN* objects and compute their average results. When calling this class, we can either use the *FreeMono* or *Inconsolata* font files to choose which font to use. It will then be saved in the corresponding directory in *data/results/<FONTNAME>*.

To compute those 30 instances of *ESN*, use the following command (with either *freemono* or *inconsolata* as argument, or nothing to select *freemono* by default):

```bash
    python main.py inconsolata
```

Note that you can also use a seed as second argument.

```bash
    python main.py inconsolata 1639617780
```

To be able to print out those same results back again, just use the *results.py* program, using the same argument (*freemono* by default).

```bash
    python results.py inconsolata
```

Finally, you can use the program *PCA.py* to compute the PCA and then display the attractors corresponding to each memory states. If the results are already computed, the program will not create new ones, so you need to delete the old ones in *data/PCA/* if you want to start again with new results. To use it, use:

```bash
    python PCA.py
```
