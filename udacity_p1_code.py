## chess project

#%%

import chess 
import chess.pgn 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sma
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn import preprocessing
from statsmodels.discrete.discrete_model import Probit
endgame_threshold = 16

#%% [markdown]
# ## this is markdown.
# smaller text
print('hallo!')


#%% 
#data preparation
exec(open("C:\\Users\\TAMM\\Desktop\\udacity\\projekt1\\udacity_p1\\data_preparation.py").read())

df = df 

#Graphics

#%% 
#sns.barplot(data = df, x=df['Result'], y=df['WhiteElo'], color = 'Blue' )
#plt.show()

#sns.countplot(data = df, x=df['Result'], palette=  ['Black', 'Grey'], order=['1-0', '1/2-1/2', '0-1'], hue = df.Strong_Player_White )
#plt.show()

#sns.histplot(data = df, x=df['PlyCount'], color = 'Blue')
#plt.show()
#sns.barplot(data = df, y=df.PlyCount.value_counts(), x=df['PlyCount'], color='Blue')
#plt.show()

#sns.barplot(data = df, x=df['Strong_Player_Result'], y=df['PlyCount'])
#plt.show()

#from https://stackoverflow.com/questions/36362624/how-to-plot-multiple-histograms-on-same-plot-with-seaborn
sns.set_theme()  # <-- This actually changes the look of plots.
plt.hist([df[df['Pieces'] <= endgame_threshold].PlyCount, df[df['Pieces'] > endgame_threshold].PlyCount], color=['r','b'], alpha=0.5)
plt.show()
#%%
##############
#General statistics

df.describe()

#How many games do we have? 
print("Number of games: ",  df.shape[0])

#How often does an endgame arise?
endgame_cnt = df[df['Pieces']<=endgame_threshold].Pieces.count()
print("Share of endgames %3.2f" %  (endgame_cnt/df.shape[0]))

#How many half-moves are played on average? 
move_cnt = df.PlyCount.mean()
print("Average Number of half-moves %3.2f" %  (move_cnt))

#How well was the outcome predicted on average?
print('We expected %3.2f points for white and got %3.2f' % (df.Probability.mean(), df.Result2.mean()+0.5) )


#Do stronger players play more moves on average?
print('Average number of moves for')
print('Stronger players: ', df[(df['BlackElo'] > 2000) | (df['WhiteElo'] > 2000)].PlyCount.mean())
print('Weaker players: ', df[(df['BlackElo'] <= 2000) | (df['WhiteElo'] <= 2000)].PlyCount.mean())


#Create dataset strong players vs weak players
strong_weak = df[((df['BlackElo'] <= 2000) & (df['WhiteElo'] > 2000)) | ((df['BlackElo'] > 2000) & (df['WhiteElo'] <= 2000))]
#print(strong_weak)


##############
#How good is the performance of the stronger players vs. weaker players?

result = strong_weak.Strong_Player_Result.mean()+.5
prob = strong_weak.Strong_Player_Prob.mean()
count = strong_weak.shape[0]

print('Number of games with strong vs weak players: ', count)
print('Avg. Points for stronger players really: ', result)
print('Avg. Points for stronger players expected: ', prob)


##############
#How good is the performance of the stronger players if an endgame was reached vs. weaker players?

result  = strong_weak[strong_weak['Pieces'] <= endgame_threshold].Strong_Player_Result.mean()+.5
prob = strong_weak[strong_weak['Pieces'] <= endgame_threshold].Strong_Player_Prob.mean()
count = strong_weak[strong_weak['Pieces'] <= endgame_threshold].shape[0]

print('Number of endgames with strong vs weak players: ', count)
print('Avg. Points for stronger players in endgame really: ', result)
print('Avg. Points for stronger players in endgame expected: ', prob)

#%%
#regression
exec(open("C:\\Users\\TAMM\\Desktop\\udacity\\projekt1\\udacity_p1\\regression.py").read())


# %%
