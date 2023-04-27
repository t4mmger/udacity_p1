##########
#Regression part

#Split into train and test


test = df['Elo_Dif']
x1 = test.values #returns a numpy array
x1 = x1.reshape(-1,1)
min_max_scaler = preprocessing.MinMaxScaler()
x1_scaled = min_max_scaler.fit_transform(x1)
df['Elo_Dif2'] = pd.DataFrame(x1_scaled)

X = df[['Elo_Dif2', 'Strong_Player_White']]
y = df['Strong_Player_Result']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .30, random_state=42)

lm_model = LinearRegression(fit_intercept=True #normalize = True
    ) # Instantiate

lm_model.fit(X_train, y_train) #Fit

y_test_preds = lm_model.predict(X_test)
y_train_preds = lm_model.predict(X_train)

test_score = r2_score(y_test, y_test_preds)
train_score = r2_score(y_train, y_train_preds)
print(test_score, train_score)

print(y_train_preds.mean(), y_train.mean())
print(y_test_preds.mean(), y_test.mean())


## from Udacity course
def coef_weights(coefficients, X_train):
    '''
    INPUT:
    coefficients - the coefficients of the linear model 
    X_train - the training data, so the column names can be used
    OUTPUT:
    coefs_df - a dataframe holding the coefficient, estimate, and abs(estimate)
    
    Provides a dataframe that can be used to understand the most influential coefficients
    in a linear model by providing the coefficient estimates along with the name of the 
    variable attached to the coefficient.
    '''
    coefs_df = pd.DataFrame()
    coefs_df['est_int'] = X_train.columns
    coefs_df['coefs'] = lm_model.coef_
    coefs_df['abs_coefs'] = np.abs(lm_model.coef_)
    coefs_df = coefs_df.sort_values('abs_coefs', ascending=False)
    return coefs_df

#Use the function
coef_df = coef_weights(lm_model.coef_, X_train)

#A quick look at the top results

#From https://www.cluzters.ai/forums/topic/395/find-p-value-significance-in-scikit-learn-linear-regression?c=1597

#X_train2 = sma.add_constant(X_train)
est = sma.OLS(y_train, X_train)
est2 = est.fit()
print(est2.summary())
#print(coef_df.head())