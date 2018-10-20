import pandas as pd
import numpy as np
from scipy.stats import poisson, skellam
import os


stats = pd.read_csv('Stats.csv')

df = pd.concat([stats[['MatchDate', 'HomeTeam', 'AwayTeam', 'Full Time Home Team Goals', 'Half Time Home Team Goals', 'Full Time Result', 'Half Time Result', 'Referee', 'Home Team Shots', 'Home Team Shots on Target', 'Home Team Fouls Committed', 'Home Team Corners', 'Home Team Yellow Cards', 'Home Team Red Cards', 'B365H', 'B365D']].assign(Home=1).rename(columns={
    'MatchDate': 'MatchDate', 'HomeTeam': 'Team', 'AwayTeam': 'Opponent', 'Full Time Home Team Goals': 'FullTimeGoals', 'Half Time Home Team Goals': 'HalfTimeGoals', 'Full TimeResult': 'FinalResult', 'Half Time Result': 'HalfTimeResult', 'Referee': 'Referee', 'Home Team Shots': 'Shots', 'Home Team Shots on Target': 'ShotsonTarget', 'Home Team FoulsCommitted': 'FoulsCommitted', 'Home Team Corners': 'Corners', 'Home Team Yellow Cards': 'YellowCards', 'Home Team Red Cards': 'RedCards', 'B365H': 'B365_Win', 'B365D': 'B365_Draw'}), stats[['MatchDate', 'AwayTeam', 'HomeTeam', 'Full Time Away Team Goals', 'Half Time Away Team Goals', 'Full Time Result', 'Half Time Result', 'Referee', 'Away Team Shots', 'Away Team Shots on Target', 'Away Team Fouls Committed', 'Away Team Corners', 'Away Team Yellow Cards', 'Away Team Red Cards', 'B365A', 'B365D']].assign(Home=0).rename(columns={
        'MatchDate': 'MatchDate', 'AwayTeam': 'Team', 'HomeTeam': 'Opponent', 'Full Time Away Team Goals': 'FullTimeGoals', 'Half Time Away Team Goals': 'HalfTimeGoals', 'Full Time Result': 'FinalResult', 'Half Time Result': 'HalfTimeResult', 'Referee': 'Referee', 'Away Team Shots': 'Shots', 'Away Team Shots on Target': 'ShotsonTarget', 'Away Team Fouls Committed': 'FoulsCommitted', 'Away Team Corners': 'Corners', 'Away Team Yellow Cards': 'YellowCards', 'Away Team Red Cards': 'RedCards', 'B365A': 'B365_Win', 'B365D': 'B365_Draw'})])


# importing the tools required for the Poisson regression model
import statsmodels.api as sm
import statsmodels.formula.api as smf
import pandas.tseries


poisson_model = smf.glm(formula="FullTimeGoals ~ Home + Team + Opponent", data=df,
                        family=sm.families.Poisson()).fit()
poisson_model.summary()

# ------------------------------------------------------------------------------------------

# Passing specific parameters into the GLM to get predicted number of goals


def match_simulation(homeTeam, awayTeam):
    homeGoals = poisson_model.predict(pd.DataFrame(
        data={'Team': homeTeam, 'Opponent': awayTeam, 'Home': 1}, index=[1]))
    awayGoals = poisson_model.predict(pd.DataFrame(
        data={'Team': awayTeam, 'Opponent': homeTeam, 'Home': 0}, index=[1]))

    return(round(homeGoals), round(awayGoals))


match_simulation('Stoke', 'Man City')




def simulate_match(foot_model, homeTeam, awayTeam, max_goals=10):
    home_goals_avg = foot_model.predict(pd.DataFrame(
        data={'Team': homeTeam, 'Opponent': awayTeam, 'Home': 1}, index=[1])).values[0]

    away_goals_avg = foot_model.predict(pd.DataFrame(
        data={'Team': awayTeam, 'Opponent': homeTeam, 'Home': 0}, index=[1])).values[0]
    
    team_pred = [[poisson.pmf(i, team_avg) for i in range(0, max_goals+1)] for team_avg in [home_goals_avg, away_goals_avg]]
    
    return(np.outer(np.array(team_pred[0]), np.array(team_pred[1])))


simulate_match(poisson_model, 'Stoke', 'Man City', max_goals=3)



game = simulate_match(poisson_model, 'Stoke', 'Man City', max_goals=10)
np.sum(np.tril(game, -1))       # Home Win
np.sum(np.diag(game))           # Draw
np.sum(np.triu(game, 1))        # Away Win


schedule = pd.read_csv('Schedule.csv')
gameweek = schedule[schedule['Round Number'] == 9]

for index, row in gameweek.iterrows():
    try:
        print(row['Home Team'], row['Away Team'], match_simulation(row['Home Team'], row['Away Team']))
    except KeyError:
        print(row['Away Team'], row['Away Team'], 'Key Error')


