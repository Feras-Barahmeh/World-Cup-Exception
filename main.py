from prepareDataset import *
from WorldCupExpectationModel import *

# Load Data
# worldCup = pd.read_csv('datasets/World Cup 2018 Dataset.csv') # For Browse

results = pd.read_csv('datasets/results.csv') # All Matches from 1872-
prepare = prepareDataset({
    "results": results
})

preparedData = prepare.prepare()

model = WorldCupExpectationModel(preparedData)


model.buildingModel()
