
1. Installations:

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

2. Project Motivations

In this project I analyse chess games with players rated between 2000 and 2500 elo rating. I do some statistics on the complete dataset before answering the following questions:

Is there a difference of playing style between players with Elo >= 2250 and players with Elo < 2250? How often do they reach endgames and how many moves to they play? Is the probability for a point suggested by Elo still valid when we enter the endgame phase? 
### Are there differences in play between weak and strong players?

If strong players play vs weaker, do we see different patterns in the game? E.g. longer games because the stronger side tries harder to win? Games that take longer until they become endgames because the stronger players try to keep more pieces on the board to outplay the opponent? 
### Are game staticts different when strong players play against weaker players? 


The Elo-difference gives a probability how many points the stronger person is expected to score. The color is not taken into account. I run a regression to check answer this question:
### Is the color White still a factor?

This analyses could prove useful for someone who would raise his rating from 2000 to > 2250. With some modifications the usefulness can be widen to a broader group. 

3. File Descsription

pgn file - 
code file - 
readme file -

4. Technical details:

I use Python 3.10.9, packaged by Anaconda Inc. 

5. Licensing, Authors, Acknowledgements 

