import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from Enums import WinnerTeam
from stringcolor import *
class WorldCupExpectationModel:
    def __init__(self, dataset):
        self.logreg = None
        self.predictionSet = []
        self.boolDataset = None
        self.dataset = dataset
        self.inputs = None
        self.output = None
        self.XTrainingData = None
        self.YTrainingData = None
        self.XTestData = None
        self.YTestData = None
        self.ranking = None
        self.fixtures = None
        self.accuracyTraining = None
        self.accuracyTesting = None

    def setInputOutputs(self, data):
        self.inputs = data.drop(["winner_team"], axis=1)
        self.output = data["winner_team"]
        self.output = self.output.astype("int")


    @staticmethod
    def oneHotEncoder(df, cols, prefix):
        """
            In this method we will represent the string category to number name country for example
            can't use it as a string data type we will represent each country to number
        :param: df the data frame object you want to apply hot encoder
        :param: cols the columns you want to convert it to boolean
        :param: prefix name new column
        :return: DataFrame object
        """
        df = pd.get_dummies(
            df,
            prefix=prefix,
            columns=cols
        )

        return df

    def separateTestAndTrainingData(self):
        self.XTrainingData, self.XTestData, self.YTrainingData, self.YTestData = train_test_split(
            self.inputs,
            self.output,
            test_size=0.30,
            random_state=42
        )

    def setFifaRanking(self):
        self.ranking = pd.read_csv("datasets/fifa_rankings.csv")

    def setFixtures(self):
        self.fixtures = pd.read_csv("datasets/fixtures.csv")
    def LR(self):
        self.logreg = LogisticRegression(max_iter=1000)
        self.logreg.fit(self.XTrainingData, self.YTrainingData)

        self.accuracyTraining = self.logreg.score(self.XTrainingData, self.YTrainingData)
        self.accuracyTesting = self.logreg.score(self.XTestData, self.YTestData)
    def setRankForEachTeam(self):
        """
            Create Column with Fifa ranking position of each team
        :return: void
        """

        self.fixtures.insert(
            1,
            "Home_rank",
            self.fixtures["Home Team"].map(self.ranking.set_index("Team")["Position"])
        )

        self.fixtures.insert(
            2,
            "Away_rank",
            self.fixtures["Away Team"].map(self.ranking.set_index("Team")["Position"])
        )

        self.fixtures = self.fixtures.iloc[:48, :] # To specific group stage

    def sortTeams(self):
        """
        Loop to add teams to new prediction dataset based on the ranking position of each team
        :return: void
        """

        for index, row in self.fixtures.iterrows():

            if row['Home_rank'] < row['Away_rank']:
                self.predictionSet.append({
                    'home_team': row['Home Team'], 'away_team': row['Away Team'], 'winning_team': None
                })
            else:
                self.predictionSet.append({
                    'home_team': row['Away Team'], 'away_team': row['Home Team'], 'winning_team': None
                })

        self.predictionSet = pd.DataFrame(self.predictionSet)

        backupPredSet = self.predictionSet


        return backupPredSet

    def addMissingCols(self):
        """
        Add missing columns compared to the model's training dataset (The match from 1932)
        in other words we will get all team founded the countries from 1932 and not found in this word cup
        :return:  void
        """

        missingCols = set(self.boolDataset.columns) - set(self.predictionSet.columns)

        newColumns = pd.DataFrame(0, index=self.predictionSet.index, columns=list(missingCols))
        self.predictionSet = pd.concat([self.predictionSet, newColumns], axis=1)

        self.predictionSet = self.predictionSet[self.boolDataset.columns]

        # Remove winning team column
        self.predictionSet = self.predictionSet.drop(['winner_team'], axis=1)

    def groupsPrediction(self, backupPredSet):
        prediction = self.logreg.predict(self.predictionSet)

        for i in range(self.fixtures.shape[0]):
            print(backupPredSet.iloc[i, 1] + " and " + backupPredSet.iloc[i, 0])
            tap = cs("-- -- ", "gray")
            if prediction[i] == WinnerTeam.HomeTeamWin.value:
                print(tap, end='')
                print(cs("Winner: ", "green"), end='')
                print(cs(backupPredSet.iloc[i, 1], "blue"))


            elif prediction[i] == WinnerTeam.Draw.value:

                print(tap, end='')
                print(cs("Draw: ", "yellow"), end='')
                print(cs(backupPredSet.iloc[i, 1], "blue"))

            elif prediction[i] == WinnerTeam.HomeTeamLoss.value:
                print(tap, end='')
                print(cs("Lose: ", "red"), end='')
                print(cs(backupPredSet.iloc[i, 0], "blue"))

            print('Probability of ' + backupPredSet.iloc[i, 1] + ' winning: ',
                  '%.3f' % (self.logreg.predict_proba(self.predictionSet)[i][2]))

            print('Probability of Draw: ', '%.3f' % (self.logreg.predict_proba(self.predictionSet)[i][1]))

            print('Probability of ' + backupPredSet.iloc[i, 0] + ' winning: ',
                  '%.3f' % (self.logreg.predict_proba(self.predictionSet)[i][0]))
            print("")

    def predict(self, stages):
        """
        To clean and predict all stage after group stage
        :param stages:
        :return:
        """

        pos = []

        # Loop to retrieve each team's position according to FIFA ranking
        for stage in stages:
            pos.append(self.ranking.loc[self.ranking['Team'] == stage[0], 'Position'].iloc[0])
            pos.append(self.ranking.loc[self.ranking['Team'] == stage[1], 'Position'].iloc[0])

    def buildingModel(self):
        self.boolDataset = self.oneHotEncoder(self.dataset, cols=["home_team", "away_team"],
                                              prefix=["home_team", "away_team"])

        self.setInputOutputs(self.boolDataset)
        self.separateTestAndTrainingData()
        self.LR()
        self.setFifaRanking()
        self.setFixtures()
        self.setRankForEachTeam()
        backupPredSet = self.sortTeams()

        self.predictionSet = self.oneHotEncoder(
            self.predictionSet, cols=["home_team", "away_team"], prefix=["home_team", "away_team"]
        )

        self.addMissingCols()
        self.groupsPrediction(backupPredSet)

        group16 = [
            ('Uruguay', 'Portugal'),
            ('France', 'Croatia'),
            ('Brazil', 'Mexico'),
            ('England', 'Colombia'),
            ('Spain', 'Russia'),
            ('Argentina', 'Peru'),
            ('Germany', 'Switzerland'),
            ('Poland', 'Belgium')
        ]

        self.predict(group16)

        