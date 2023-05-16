import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as ticker
import matplotlib.ticker as plticker
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from prepareDataset import *

# Load Data
# worldCup = pd.read_csv('datasets/World Cup 2018 Dataset.csv') # For Browse

results = pd.read_csv('datasets/results.csv') # All Matches from 1872-
prepare = prepareDataset({
    "results": results
})

preparedData = prepare.prepare()



