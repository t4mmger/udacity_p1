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

    # this collects the headers information
    for el in headers: 
        headers[el] += [g.headers[el]]
    
    # this collects the number of pieces at the end of the game
    pieces['Pieces'] += [32 - str(g.mainline_moves()).count('x')]
    if i >= 5:
        break

chess_df = {**headers,**pieces}

df = pd.DataFrame(chess_df)

df[['WhiteElo','BlackElo','Pieces','PlyCount']] = df[['WhiteElo', 'BlackElo','Pieces','PlyCount']].apply(pd.to_numeric)



#https://stackoverflow.com/questions/34962104/how-can-i-use-the-apply-function-for-a-single-column
#def elo_prob(w,b):
    #print('W ist ', w, '. B ist ', b)
    #if w >= b:
    #    print('wir sind im if, we return ', 1 / (1+10**((w - b)/400) ))
    #return 1 / (1+10**(-(w - b)/400))
    #    
    #else:
    #    print('wir sind im else')
    #    return 1 / (1+10**(-(w - b)/400))

#def apply_elo_prob(x):
 #   return elo_prob(x['WhiteElo'],x['BlackElo'])

#df['Probability'] = df.apply(apply_elo_prob, axis = 1)    


#formula for probability from https://fivethirtyeight.com/features/introducing-nfl-elo-ratings/
df['Probability'] = 1 / (1+10**(-(df['WhiteElo'] - df['BlackElo'])/400))

df.Result2 = df.Result.apply(lambda val : 1 if val == '1-0' else (0 if val == '0-1' else 0.5))


#How often does an endgame arise?
endgame_cnt = df[df['Pieces']<=16].Pieces.count()
print("Share of endgames %3.2f" %  (endgame_cnt/df.shape[0]))

#How many half-moves are played on average? 
move_cnt = df.PlyCount.mean()
print("Average Number of moves %3.2f" %  (move_cnt))

#How well was the outcome predicted on average?
print('We expected %3.2f points for white and got %3.2f' % (df.Probability.mean(), df.Result2.mean()) )


#Do stronger players play more moves on average?
print(df[(df['BlackElo'] > 2000) | (df['WhiteElo'] > 2000)].head)

#How good is the performance of the stronger players if an endgame was reached?


