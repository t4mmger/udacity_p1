
#### This code does not run on its own. It should be called by udacity_p1_code.py. 


#set path and thresholds
pgn = open("C:\\Users\\TAMM\\Desktop\\udacity\\projekt1\\1980_1989.pgn", encoding= 'unicode_escape')

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
others = {'Pieces': []
#, 'PlyCount2': []
}
for i, g in enumerate(my_games):
    if i in (10000, 20000, 30000, 40000, 50000, 60000, 70000, 80000, 90000, 100000, 110000 , 120000, 130000): 
        print(i)
        
        #https://www.programiz.com/python-programming/datetime/current-datetime
        # datetime object containing current date and time
        now = datetime.now()
        # dd/mm/YY H:M:S
        dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
        print("date and time =", dt_string)

    else: 
        None 
    
    # this collects the headers information
    for el in headers: 
        try: 
            headers[el] += [g.headers[el]]
        except: 
            if el == 'PlyCount':
                try: 
                    moves = 2*max(eval(j) for j in re.findall(r'\b\d+\b', str(g.mainline_moves())))
                    headers[el] += [moves]
                except:
                    headers[el] += ['0']
            else: 
                headers[el] += ['0']
    # this collects the number of pieces at the end of the game
    others['Pieces'] += [32 - str(g.mainline_moves()).count('x')]

    #others['PlyCount2'] += [moves]

#This helped me with the loop: 
#https://www.geeksforgeeks.org/python-converting-all-strings-in-list-to-integers/
#https://stackoverflow.com/questions/4289331/how-to-extract-numbers-from-a-string-in-python

#for key, value in chess_df.items():
#   #print value
#   print(key, len([item for item in value if item])) 

chess_df = {**headers,**others}
df = pd.DataFrame(chess_df)
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

#Data Cleaning
df = pd.DataFrame.drop_duplicates(df)
df = df[(df['PlyCount'] > 0) & (df['WhiteElo'] > 0) & (df['BlackElo'] > 0)]

#Save dataset so we do not need to create it again
df.to_csv(path_or_buf='C:\\Users\\TAMM\\Desktop\\udacity\\projekt1\\udacity_p1\\game_result_list.csv', sep=';', na_rep='', float_format=None, columns=None, header=True, index=False, index_label=None, mode='w', encoding='utf-8', compression='infer', quoting=None, quotechar='"', line_terminator=None, chunksize=None, date_format=None, doublequote=True, escapechar=None, decimal='.')

print('Fertig!')
# %%
