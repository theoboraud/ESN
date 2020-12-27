"""
@author: Theophile BORAUD
t.boraud@warwick.co.uk
Copyright 2019, Theophile BORAUD, Anthony STROCK, All rights reserved.
"""


import os


os.system('python3 main.py freemono 1639617780')
os.system('python3 main.py inconsolata 3939310522')
os.system('rm data/PCA/freemono/*')
os.system('python3 PCA.py')
