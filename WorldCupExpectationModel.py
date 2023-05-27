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

    def addMissingCols(self, df):
        """
        Add missing columns compared to the model's training dataset (The match from 1932)
        in other words we will get all team founded the countries from 1932 and not found in this word cup
        :param: data frame object you want find missing columns
        :return:  DataFrame
        """

        missingCols = set(self.boolDataset.columns) - set(df.columns)

        newColumns = pd.DataFrame(0, index=df.index, columns=list(missingCols))
        df = pd.concat([df, newColumns], axis=1)

        df = df[self.boolDataset.columns]

        # Remove winning team column
        df = df.drop(['winner_team'], axis=1)

        return df

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

    def cleanBeforePredict(self, stages):
        """
        To clean Data Before predict
        :param stages:
        :return: clean prediction data frame and backup Predicted data frame
        """
        pos = []

        # Loop to retrieve each team's position according to FIFA ranking
        for stage in stages:
            pos.append(self.ranking.loc[self.ranking['Team'] == stage[0], 'Position'].iloc[0])
            pos.append(self.ranking.loc[self.ranking['Team'] == stage[1], 'Position'].iloc[0])

            # Creating the DataFrame for prediction
        pred = []

        # Initializing iterators for while loop
        i = 0
        j = 0

        # 'i' will be the iterator for the 'positions' list, and 'j' for the list of matches (list of tuples)
        while i < len(pos):
            dictTemp = {}

            # If position of first team is better, he will be the 'home' team, and vice-versa
            if pos[i] < pos[i + 1]:
                dictTemp.update({'home_team': stages[j][0], 'away_team': stages[j][1]})
            else:
                dictTemp.update({'home_team': stages[j][1], 'away_team': stages[j][0]})

            # Append updated dictionary to the list, that will later be converted into a DataFrame
            pred.append(dictTemp)
            i += 2
            j += 1

        # Convert list into DataFrame
        pred = pd.DataFrame(pred)
        backupPred = pred

        return pred, backupPred

    def predict(self, stages, messageSeparator):
        """
        To predict all stage after group stage
        :param messageSeparator:
        :param stages:
        :return:
        """
        pred, backupPred = self.cleanBeforePredict(stages)

        # Get dummy variables and drop winning_team column
        pred = self.oneHotEncoder(pred, cols=['home_team', 'away_team'], prefix=['home_team', 'away_team'])

        pred = self.addMissingCols(pred)

        # Predict!
        messageSeparator = '-' * 10 + ' ' + messageSeparator + ' ' + '-' * 10
        print(cs(messageSeparator, "green"), end='\n\n')

        predictions = self.logreg.predict(pred)

        for i in range(len(pred)):
            print(backupPred.iloc[i, 1] + " and " + backupPred.iloc[i, 0])
            if predictions[i] == 2:
                print(cs("Winner: ", "green") + backupPred.iloc[i, 1])
            elif predictions[i] == 1:
                print(cs("Winner: ", "yellow"))
            elif predictions[i] == 0:
                print(cs("Winner: ", "green") + backupPred.iloc[i, 0])


            print('Probability of ' + backupPred.iloc[i, 1] + ' winning: ',
                  '%.3f' % (self.logreg.predict_proba(pred)[i][2]))

            print('Probability of Draw: ', '%.3f' % (self.logreg.predict_proba(pred)[i][1]))
            print('Probability of ' + backupPred.iloc[i, 0] + ' winning: ',
                  '%.3f' % (self.logreg.predict_proba(pred)[i][0]))
            print("")




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

        self.predictionSet = self.addMissingCols(self.predictionSet)
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

        self.predict(group16, "Stage 16")

        quarters = [('Portugal', 'France'),
                    ('Spain', 'Argentina'),
                    ('Brazil', 'England'),
                    ('Germany', 'Belgium')]

        self.predict(quarters, "Stage quarters")

        semi = [('Portugal', 'Brazil'),
                ('Argentina', 'Germany')]

        self.predict(semi, "Stage semi final")

        finals = [('Brazil', 'Germany')]

        self.predict(finals, "finals")