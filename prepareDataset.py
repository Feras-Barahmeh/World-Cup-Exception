import pandas as pd
import numpy as np
from Enums import WinnerTeam

class prepareDataset:
    def __init__(self, files: dict):
        self.files = files
        self.worldCupTeams = None
        self.teams = None

    def establishingWinnerTeams(self):
        """
            In this method we will add new column in results dataset to determine which team is wen for each match
        :return: void
        """
        winners = []
        for match in range(len(self.files["results"]["home_team"])):
            if self.files["results"]["home_score"][match] > self.files["results"]["away_score"][match]:
                winners.append(self.files["results"]["home_team"][match])
            elif self.files["results"]["home_score"][match] < self.files["results"]["away_score"][match]:
                winners.append(self.files["results"]["away_team"][match])
            else:
                winners.append("Draw")

        self.files["results"]["winner_team"] = winners

    def addDifferenceGoalsForEachMatch(self):
        """
            This Method To Add New Column in results dataset this colum represent the difference goals for each match
        :return: void
        """
        self.files["results"]["goal_diff"] = np.absolute(
            self.files["results"]["home_score"] - self.files["results"]["away_score"]
        )

    def appendFileInFiles(self, nameFile, file):
        """
            To add new file in files attribute
        :param nameFile: Name File as key
        :param file: dataset
        :return: void
        """
        self.files.update({nameFile: file})

    def setWorldCupTeam(self):
        """
            To set all world in cup
        :return: void
        """
        self.worldCupTeams = [
            'Australia', ' Iran', 'Japan', 'Korea Republic',
            'Saudi Arabia', 'Egypt', 'Morocco', 'Nigeria',
            'Senegal', 'Tunisia', 'Costa Rica', 'Mexico',
            'Panama', 'Argentina', 'Brazil', 'Colombia',
            'Peru', 'Uruguay', 'Belgium', 'Croatia',
            'Denmark', 'England', 'France', 'Germany',
            'Iceland', 'Poland', 'Portugal', 'Russia',
            'Serbia', 'Spain', 'Sweden', 'Switzerland']

    def narrowingToTeamParticipatingInTheWorldCup(self):
        homeTeams = self.files["results"][self.files["results"]["home_team"].isin(self.worldCupTeams)]
        awayTeams = self.files["results"][self.files["results"]["away_team"].isin(self.worldCupTeams)]
        self.teams = pd.concat((homeTeams, awayTeams))
        self.teams.drop_duplicates()

    def dropMatchBefore1930(self):
        years = []
        for row in self.teams["date"]:
            years.append(int(row[:4]))

        self.teams["match_year"] = years
        self.teams = self.teams[self.teams.match_year >= 1930]


    def dropNotAffectiveColumns(self):
        """
            Drop all columns not affective in model (pass misleading)
        :return: void
        """
        self.teams = self.teams.drop(
            [
                "date",
                "home_score",
                "away_score",
                "tournament",
                "city",
                "country",
                "goal_diff",
                "match_year"
            ],
            axis=1
        )


    def showResults(self, numberHeads=None):
        if not numberHeads:
            print(self.files["results"].head())
        else:
            print(self.files["results"].head(numberHeads))


    def prepare(self):
        self.establishingWinnerTeams()
        self.addDifferenceGoalsForEachMatch()
        self.setWorldCupTeam()
        self.narrowingToTeamParticipatingInTheWorldCup()
        self.dropMatchBefore1930()
        self.dropNotAffectiveColumns()
        return self.teams