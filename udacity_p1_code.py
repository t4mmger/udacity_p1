## chess project

import pandas as pd
import chess 
import chess.pgn 

pgn = open("C:\\Users\\TAMM\\Desktop\\udacity\\projekt1\\kem_mfrmitte_2023.pgn")

my_games = []


#from https://github.com/niklasf/python-chess/issues/446
while True:
    game = chess.pgn.read_game(pgn)
    if game is None:
        break  # end of file
    my_games.append(game)
####


headers = {'White': [], 'WhiteElo': [], 'Black': [], 'BlackElo': [], 'Result': [], 'ECO': [], 'PlyCount': [], 'Date': []}
pieces = {'Pieces': []}
for i, g in enumerate(my_games):
    print(i)
    # this collects the headers information
    for el in headers: 
        print(el)
        print(g.headers[el])
        headers[el] += [g.headers[el]]
    
    # this collects the number of pieces at the end of the game
    pieces['Pieces'] += [32 - str(g.mainline_moves()).count('x')]
    if i >= 2:
        break

chess_df = {**headers,**pieces}

#print(my_games[0].mainline_moves())

df = pd.DataFrame(chess_df)

df[['WhiteElo','BlackElo']] = df[['WhiteElo', 'BlackElo']].apply(pd.to_numeric)

#formula for probability from https://fivethirtyeight.com/features/introducing-nfl-elo-ratings/
df['Probability'] = 1 / (1+10**((df['WhiteElo'] - df['BlackElo'])/400))

print(df)
df.dtypes

print(my_games[1])