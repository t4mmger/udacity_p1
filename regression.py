##########
#Regression part

#Split into train and test

X = df[['Elo_Dif', 'Strong_Player_White']]
y = df['Strong_Player_Result']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .30, random_state=42)

lm_model = LinearRegression(fit_intercept=True #normalize = True
    ) # Instantiate

lm_model.fit(X_train, y_train) #Fit

y_test_preds = lm_model.predict(X_test)
y_train_preds = lm_model.predict(X_train)

test_score = r2_score(y_test, y_test_preds)
train_score = r2_score(y_train, y_train_preds)
print('R^2 Score is on our test sample is: %5.4f' % (test_score))
print('In Training we had: %5.4f' % (train_score))

#print('Mean predictions and real mean on traing set:  %5.4f vs %5.4f' %  (y_train_preds.mean(), y_train.mean()))
print('Mean predictions and real mean on test set: %5.4f vs %5.4f' % ( y_test_preds.mean(), y_test.mean()))

#From https://www.cluzters.ai/forums/topic/395/find-p-value-significance-in-scikit-learn-linear-regression?c=1597

X_train2 = sma.add_constant(X_train)
est = sma.OLS(y_train, X_train2)
est2 = est.fit()
print(est2.summary())

#Probit might be better than linear
#from https://jbhender.github.io/Stats506/F18/GP/Group14.html#python
#model = Probit(y_train, X_train2)
#probit_model = model.fit()
#print(probit_model.summary())