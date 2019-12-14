import csv
import pandas as pd
import numpy as np
import requests
import datetime
from scipy.stats import poisson
import statsmodels.api as sm
import statsmodels.formula.api as smf

stats_columns = ['MatchDate', 'HomeTeam', 'AwayTeam', 'Full Time Home Team Goals', 'Half Time Home Team Goals',
                 'Full Time Result', 'Half Time Result', 'Referee', 'Home Team Shots', 'Home Team Shots on Target',
                 'Home Team Fouls Committed', 'Home Team Corners', 'Home Team Yellow Cards', 'Home Team Red Cards',
                 'B365H', 'B365D']


def current_season():
    now = datetime.datetime.now()
    if now.month in (8, 9, 10, 11, 12):
        return str(now.year)[-2:] + str(now.year + 1)[-2:]
    elif now.month in (1, 2, 3, 4, 5):
        return str(now.year - 1)[-2:] + str(now.year)[-2:]
    else:
        return str(now.year - 1)[-2:] + str(now.year)[-2:]


def get_latest_results():
    url = f'https://www.football-data.co.uk/mmz4281/{current_season()}/E0.csv'
    with requests.Session() as s:
        download = s.get(url)
        decoded_content = download.content.decode('utf-8')
        cr = csv.reader(decoded_content.splitlines(), delimiter=',')
        my_list = list(cr)

    all_games = pd.DataFrame(data=my_list)
    all_games = all_games[1:]  # take the data less the header row
    all_games.columns = all_games.iloc[0]  # set the header row as the df header


def parse_games_file():
    stats = pd.read_csv('Stats.csv')

    df = pd.concat(
        [stats[stats_columns].assign(Home=1).rename(columns={
            'MatchDate': 'MatchDate', 'HomeTeam': 'Team', 'AwayTeam': 'Opponent',
            'Full Time Home Team Goals': 'FullTimeGoals', 'Half Time Home Team Goals': 'HalfTimeGoals',
            'Full TimeResult': 'FinalResult', 'Half Time Result': 'HalfTimeResult', 'Referee': 'Referee',
            'Home Team Shots': 'Shots', 'Home Team Shots on Target': 'ShotsonTarget',
            'Home Team FoulsCommitted': 'FoulsCommitted', 'Home Team Corners': 'Corners',
            'Home Team Yellow Cards': 'YellowCards', 'Home Team Red Cards': 'RedCards', 'B365H': 'B365_Win',
            'B365D': 'B365_Draw'}),
            stats[stats_columns].assign(Home=0).rename(columns={
                'MatchDate': 'MatchDate', 'AwayTeam': 'Team', 'HomeTeam': 'Opponent',
                'Full Time Away Team Goals': 'FullTimeGoals', 'Half Time Away Team Goals': 'HalfTimeGoals',
                'Full Time Result': 'FinalResult', 'Half Time Result': 'HalfTimeResult', 'Referee': 'Referee',
                'Away Team Shots': 'Shots', 'Away Team Shots on Target': 'ShotsonTarget',
                'Away Team Fouls Committed': 'FoulsCommitted', 'Away Team Corners': 'Corners',
                'Away Team Yellow Cards': 'YellowCards', 'Away Team Red Cards': 'RedCards', 'B365A': 'B365_Win',
                'B365D': 'B365_Draw'})
        ], sort=True)

    return df


def model_data(df):
    return smf.glm(formula="FullTimeGoals ~ Home + Team + Opponent", data=df,
                   family=sm.families.Poisson()).fit()


def match_simulation(df, home_team, away_team, max_goals=3, verbose=0):
    poisson_model = model_data(df)

    home_goals = poisson_model.predict(
        pd.DataFrame(data={'Team': home_team, 'Opponent': away_team, 'Home': 1}, index=[1])
    ).values[0]

    away_goals = poisson_model.predict(
        pd.DataFrame(data={'Team': away_team, 'Opponent': home_team, 'Home': 0}, index=[1])
    ).values[0]

    print(f'{home_team}: {round(home_goals, 2)}; {away_team}: {round(away_goals, 2)}')

    team_prediction = [
        [poisson.pmf(i, team_avg) for i in range(0, max_goals + 1)] for team_avg in [home_goals, away_goals]
    ]

    game = np.outer(np.array(team_prediction[0]), np.array(team_prediction[1]))

    home = np.sum(np.tril(game, -1))
    draw = np.sum(np.diag(game))
    away = np.sum(np.triu(game, 1))

    if verbose == 1:
        print(f'Home: {home}; Draw: {draw}; Away: {away}')


def main():
    df = parse_games_file()

    schedule = pd.read_csv('Schedule.csv')
    game_week = schedule[schedule['Round Number'] == 9]

    for index, row in game_week.iterrows():
        home = row['Home Team']
        away = row['Away Team']
        match_simulation(df, home, away)


if __name__ == '__main__':
    main()
