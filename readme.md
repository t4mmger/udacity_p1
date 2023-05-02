
# 1. Installations:

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

# 2. Project Motivations

In this project I analyse chess games with players rated between 2000 and 2500 elo rating. I do some statistics on the complete dataset before answering the following questions:

Is there a difference of playing style between players with Elo >= 2250 and players with Elo < 2250? How often do they reach endgames and how many moves to they play? Is the probability for a point suggested by Elo still valid when we enter the endgame phase? 
### Are there differences in play between weak and strong players?

If strong players play vs weaker, do we see different patterns in the game? E.g. longer games because the stronger side tries harder to win? 
### Are game statistics different when strong players play against weaker players? 


The Elo-difference gives a probability how many points the stronger person is expected to score. The color is not taken into account. I run a regression to check answer this question:
### Is the color White still a factor?

This analyses could prove useful for someone who would raise his rating from 2000 to > 2250. With some modifications the usefulness can be widen to a broader group. 

# 3. File Descsription

pgn file: 1980_1989.pgn a sample of chess games from 1980-1989 with ratings between 2000 and 2500. \
csv file: game_result_list.csv, selected information from the pgn file and also some cleaning was done. see data_preparation.py for details. \
py files: data_preparation.py the data steps to transform the pgn into csv. \ udacity_p1_code.py main file with import of modules, statistics and regression.
readme file: 

# 4. Technical details:

I use Python 3.10.9, packaged by Anaconda Inc. 

# 5. Licensing, Authors, Acknowledgements 

Author: t4mmger
Acknowledgements: to all who contribute to the python community.
Licensing: free