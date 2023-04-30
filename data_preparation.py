
#### This code does not run on its own. It should be called by udacity_p1_code.py. 

#set path and thresholds
pgn = open("C:\\Users\\TAMM\\Desktop\\udacity\\projekt1\\kem_mfrmitte_2023_.pgn", encoding= 'unicode_escape')


my_games = []

#from https://github.com/niklasf/python-chess/issues/446
while True:
    game = chess.pgn.read_game(pgn)
    if game is None:
        break  # end of file
    my_games.append(game)
####
##, 'ECO': []

headers = {'White': [], 'WhiteElo': [], 'Black': [], 'BlackElo': [], 'Result': [], 'PlyCount': [], 'Date': []}
pieces = {'Pieces': []}
for i, g in enumerate(my_games):

    # this collects the headers information
    for el in headers: 
        headers[el] += [g.headers[el]]
    #print(g)
    # this collects the number of pieces at the end of the game
    pieces['Pieces'] += [32 - str(g.mainline_moves()).count('x')]
    #if i >= 10:
     #   break

chess_df = {**headers,**pieces}
df = pd.DataFrame(chess_df)
#print(df)
df[['WhiteElo','BlackElo','Pieces','PlyCount']] = df[['WhiteElo', 'BlackElo','Pieces','PlyCount']].apply(pd.to_numeric)


#formula for probability from https://fivethirtyeight.com/features/introducing-nfl-elo-ratings/
df['Probability'] = 1 / (1+10**(-(df['WhiteElo'] - df['BlackElo'])/400))
df['Result2'] = df.Result.apply(lambda val : 0.5 if val == '1-0' else (-0.5 if val == '0-1' else 0.0))

#https://stackoverflow.com/questions/34962104/how-can-i-use-the-apply-function-for-a-single-column
def result_change(w,b,r):
    if w >= b:
        return r
    else:
        return -r

def apply_rc(x):
    return result_change(x['WhiteElo'], x['BlackElo'], x['Result2'])


def prob_change(w,b,p):
    if w >= b:
        return p
    else:
        return 1-p

def apply_pc(x):
    return prob_change(x['WhiteElo'], x['BlackElo'], x['Probability'])

def color_funk(w,b):
    if w >= b:
        return 1
    else:
        return 0
    
def apply_cf(x):
    return color_funk(x['WhiteElo'], x['BlackElo'])
    
df['Strong_Player_Result'] = df.apply(apply_rc, axis=1)+0.5
df['Strong_Player_Prob'] = df.apply(apply_pc, axis=1)
df['Strong_Player_White'] = df.apply(apply_cf, axis=1)
df['Elo_Dif'] = abs(df.WhiteElo - df.BlackElo)

print('Fertig!')
# %%
