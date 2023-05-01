## chess project

#%%

import chess 
import chess.pgn 
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import re
import statsmodels.api as sma
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn import preprocessing
from statsmodels.discrete.discrete_model import Probit
endgame_threshold = 16
elo_threshold = 2250


#%% [markdown]

### Introduction

# What this analysis is about.
# Some words on the dataset. 


#%% 
#Data Preparation

#This takes some time. You don't have to run it, but import the csv file. 
#exec(open("C:\\Users\\TAMM\\Desktop\\udacity\\projekt1\\udacity_p1\\data_preparation.py").read())

df = pd.read_csv(filepath_or_buffer = "C:\\Users\\TAMM\\Desktop\\udacity\\projekt1\\udacity_p1\\game_result_list.csv", sep = ';')

#%% 

sns.countplot(data = df, x=df['Result'], palette = ['Black', 'Grey'], order=['1-0', '1/2-1/2', '0-1'], hue = df.Strong_Player_White )
plt.show()

#sns.countplot(data = df, x=df['Strong_Player_Result'], hue = df.Strong_Player_White )
#plt.show()

sns.kdeplot(data=df, x=df['PlyCount'], legend = False, hue= df['Pieces'] <= endgame_threshold, shade=True)
plt.title('How many moves were played and was an endgame reached?')
plt.legend(title = 'Endgame reached' , loc='upper right', labels=['Yes', 'No'])
plt.xlim(0,200)
plt.show()

#%% [markdown]
##############
### General statistics

print(df.describe())

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
print('Stronger players: ', df[(df['BlackElo'] > elo_threshold) | (df['WhiteElo'] > elo_threshold)].PlyCount.mean())
print('Weaker players: ', df[(df['BlackElo'] <= elo_threshold) | (df['WhiteElo'] <= elo_threshold)].PlyCount.mean())


#Create dataset strong players vs weak players
strong_weak = df[((df['BlackElo'] <= elo_threshold) & (df['WhiteElo'] > elo_threshold)) | ((df['BlackElo'] > elo_threshold) & (df['WhiteElo'] <= elo_threshold))]

##############
#How good is the performance of the stronger players vs. weaker players?

result = strong_weak.Strong_Player_Result.mean()
prob = strong_weak.Strong_Player_Prob.mean()
count = strong_weak.shape[0]

print('Number of games with strong vs weak players: ', count)
print('Avg. Points for stronger players really: ', result)
print('Avg. Points for stronger players expected: ', prob)


##############
#How good is the performance of the stronger players if an endgame was reached vs. weaker players?

result  = strong_weak[strong_weak['Pieces'] <= endgame_threshold].Strong_Player_Result.mean()
prob = strong_weak[strong_weak['Pieces'] <= endgame_threshold].Strong_Player_Prob.mean()
count = strong_weak[strong_weak['Pieces'] <= endgame_threshold].shape[0]

print('Number of endgames with strong vs weak players: ', count)
print('Avg. Points for stronger players in endgame really: ', result)
print('Avg. Points for stronger players in endgame expected: ', prob)

#%% [markdown]
# 
### Is White important for the win probability? 
# We will answer this question with a OLS regression. The y variable will be the result from the stronger player's view.
# Our X-variables will be the Elo Difference and a White-Dummy.

exec(open("C:\\Users\\TAMM\\Desktop\\udacity\\projekt1\\udacity_p1\\regression.py").read())

#%%[markdown] Indeed the white pieces are favorable. 

## How can we improve things?
#  We only did an OLS regression, but the win probability based on Elo difference follows a normal or logit distribution, so a logit or probit regression would be better.

# What do you think is most important for improving your chess?

# %%
