"""
    Kushagra Rastogi
    304640248
    ECE 219
    Project 5
"""

import numpy as np
import pandas as pd
import nltk, string
import matplotlib.pyplot as plt
import json
import datetime, time
import pytz


# hashtags = tweets_#gohawks.txt, tweets_#gopatriots.txt, tweets_#nfl.txt, tweets_#patriots.txt, tweets_#sb49.txt, tweets_#superbowl.txt

############################################################################### QUESTION 1
"""
tweetfile = 'tweets_#superbowl.txt'
file = []
with open(tweetfile,encoding="utf8") as f:
    for line in f:
        file.append(line)

hours,num_follow,num_retweets = [],[],[]
num_tweets = 0
for line in file:
    tw = json.loads(line)
    hours.append(tw['citation_date'])
    num_follow.append(tw['author']['followers'])
    num_retweets.append(tw['metrics']['citations']['total'])
    num_tweets += 1

print("Statistics for ",tweetfile)
print("Average number of tweets per hour: ", num_tweets/((max(hours)-min(hours))/3600.))
print("Average number of followers of users posting the tweets per tweet: ", sum(num_follow)/float(num_tweets) )
print("Average number of retweets per tweet: ", sum(num_retweets)/float(num_tweets)  )
"""

############################################################################### QUESTION 2 
"""
tweets_hr = [0] * int((max(hours)-min(hours))/3600+1)
start = min(hours)
for i in hours:
    tweets_hr[int((i-start)/3600)] += 1

x = [i for i in range(0,len(tweets_hr))]
plt.bar(x,tweets_hr,1)
plt.xlabel("Hour"); plt.ylabel("Number of tweets"); plt.title(tweetfile)
plt.show()
"""

############################################################################### QUESTION 3 
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error
"""
tweetfile = 'tweets_#gohawks.txt'
file = []
with open(tweetfile,encoding="utf8") as f:
    for line in f:
        file.append(line)

date,time,num_tweets,hours,num_follow,num_retweets = [],[],[],[],[],[]
for n,line in enumerate(file):
    tw = json.loads(line)
    num_tweets.append(1)
    hours.append(tw['citation_date'])
    num_follow.append(tw['author']['followers'])
    num_retweets.append(tw['metrics']['citations']['total'])

df = pd.DataFrame({'tweets':num_tweets,'hours':hours,'followers':num_follow,'retweets':num_retweets},columns=['tweets', 'hours', 'followers', 'retweets'])

pst_tz = pytz.timezone('America/Los_Angeles')
for hour in df['hours']:
    timestamp = str(datetime.datetime.fromtimestamp(int(hour), pst_tz))
    time_split = ''.join(timestamp[0:10].split('-'))
    date.append(int(time_split))
    time.append(int(timestamp[11:13]))

df.insert(1,'date',date)
df.insert(2,'time',time)
df.insert(3,'max_followers',df['followers'])
df.drop('hours', 1, inplace = True)
df1 = df.groupby(['date', 'time']).agg({'time' : np.max, 'tweets' : np.sum, 'followers' : np.sum, 'retweets' : np.sum, 'max_followers' : np.max})
X = []
for i in df1.index:
    X.append(df1.loc[i, 'time':'max_followers'].values)
X.pop()
X = sm.add_constant(X)
y = df1.loc[df1.index[1]:, 'tweets'].values
model = sm.OLS(y,X).fit()
pred_y = model.predict(X)
print("MSE for %s : %.6f" % (tweetfile, mean_squared_error(y,pred_y)))
print(model.summary())
"""

############################################################################### QUESTION 4 
"""
tweetfile = 'tweets_#superbowl.txt'
file = []
with open(tweetfile,encoding="utf8") as f:
    for line in f:
        file.append(line)

pst_tz = pytz.timezone('America/Los_Angeles')
date,time,num_tweets,num_follow,num_retweets,num_mentions,rank_score,num_hashtags = [],[],[],[],[],[],[],[]
for n,line in enumerate(file):
    tw = json.loads(line)
    timestamp = tw['citation_date']
    timestamp = str(datetime.datetime.fromtimestamp(int(timestamp), pst_tz))
    time_split = ''.join(timestamp[0:10].split('-'))
    date.append(int(time_split))
    time.append(int(timestamp[11:13]))
    
    num_tweets.append(1)
    num_follow.append(tw['author']['followers'])
    num_retweets.append(tw['metrics']['citations']['total'])
    num_mentions.append(len(tw['tweet']['entities']['user_mentions']))
    rank_score.append(tw['metrics']['ranking_score'])
    num_hashtags.append(tw['title'].count('#'))

df = pd.DataFrame({'date':date,'time':time,'tweets':num_tweets,'followers':num_follow,'retweets':num_retweets,'mentions':num_mentions,'score':rank_score,'hashtags':num_hashtags},
                  columns=['date','time','tweets', 'followers', 'retweets','mentions','score','hashtags'])

df1 = df.groupby(['date', 'time']).agg({'time' : np.max, 'tweets' : np.sum,  'followers' : np.sum, 'retweets' : np.sum, 'mentions' : np.sum, 'score' : np.sum, 'hashtags' : np.sum})
X = []
for i in df1.index:
    X.append(df1.loc[i, 'tweets':'hashtags'].values)
X.pop()
X = sm.add_constant(X)
y = df1.loc[df1.index[1]:, 'tweets'].values
model = sm.OLS(y,X).fit()
pred_y = model.predict(X)
print("MSE for %s : %.6f" % (tweetfile, mean_squared_error(y,pred_y)))
print(model.summary())
"""

############################################################################### QUESTION 5 
"""
tweetfile = 'tweets_#superbowl.txt'
file = []
with open(tweetfile,encoding="utf8") as f:
    for line in f:
        file.append(line)

pst_tz = pytz.timezone('America/Los_Angeles')
date,time,num_tweets,num_follow,num_retweets,num_mentions,rank_score,num_hashtags = [],[],[],[],[],[],[],[]
for n,line in enumerate(file):
    tw = json.loads(line)
    timestamp = tw['citation_date']
    timestamp = str(datetime.datetime.fromtimestamp(int(timestamp), pst_tz))
    time_split = ''.join(timestamp[0:10].split('-'))
    date.append(int(time_split))
    time.append(int(timestamp[11:13]))
    
    num_tweets.append(1)
    num_follow.append(tw['author']['followers'])
    num_retweets.append(tw['metrics']['citations']['total'])
    num_mentions.append(len(tw['tweet']['entities']['user_mentions']))
    rank_score.append(tw['metrics']['ranking_score'])
    num_hashtags.append(tw['title'].count('#'))

df = pd.DataFrame({'date':date,'time':time,'tweets':num_tweets,'followers':num_follow,'retweets':num_retweets,'mentions':num_mentions,'score':rank_score,'hashtags':num_hashtags},
                  columns=['date','time','tweets', 'followers', 'retweets','mentions','score','hashtags'])

df1 = df.groupby(['date', 'time']).agg({'time' : np.max, 'tweets' : np.sum,  'followers' : np.sum, 'retweets' : np.sum, 'mentions' : np.sum, 'score' : np.sum, 'hashtags' : np.sum})

feat = 'retweets'
X = df1[feat].values
X = sm.add_constant(X)
y = df1['tweets'].values
model = sm.OLS(y,X).fit()
pred_y = model.predict(X)
plt.figure()
plt.plot(y,pred_y,'bo')
plt.xlabel(feat); plt.ylabel("Tweets"); plt.title(tweetfile)
plt.show()

feat = 'score'
X = df1[feat].values
X = sm.add_constant(X)
y = df1['tweets'].values
model = sm.OLS(y,X).fit()
pred_y = model.predict(X)
plt.figure()
plt.plot(y,pred_y,'bo')
plt.xlabel(feat); plt.ylabel("Tweets"); plt.title(tweetfile)
plt.show()

feat = 'hashtags'
X = df1[feat].values
X = sm.add_constant(X)
y = df1['tweets'].values
model = sm.OLS(y,X).fit()
pred_y = model.predict(X)
plt.figure()
plt.plot(y,pred_y,'bo')
plt.xlabel(feat); plt.ylabel("Tweets"); plt.title(tweetfile)
plt.show()
"""

############################################################################### QUESTION 6 
import warnings
warnings.filterwarnings("ignore")
"""
tweetfile = 'tweets_#gohawks.txt'
file = []
with open(tweetfile,encoding="utf8") as f:
    for line in f:
        file.append(line)

pst_tz = pytz.timezone('America/Los_Angeles')
date,time,num_tweets,num_follow,num_retweets,num_mentions,rank_score,num_hashtags = [],[],[],[],[],[],[],[]
time_5  =[]
for n,line in enumerate(file):
    tw = json.loads(line)
    timestamp = tw['citation_date']
    timestamp = str(datetime.datetime.fromtimestamp(int(timestamp), pst_tz))
    time_split = ''.join(timestamp[0:10].split('-'))
    date.append(int(time_split))
    time.append(int(timestamp[11:13]))
    
    a,b,c,_ = timestamp.split(':')
    rounded = str(int((int(b)/5))*5).zfill(2)
    time_mins = timestamp[11:13] + rounded
    time_5.append(int(time_mins))
    
    num_tweets.append(1)
    num_follow.append(tw['author']['followers'])
    num_retweets.append(tw['metrics']['citations']['total'])
    num_mentions.append(len(tw['tweet']['entities']['user_mentions']))
    rank_score.append(tw['metrics']['ranking_score'])
    num_hashtags.append(tw['title'].count('#'))

df = pd.DataFrame({'date':date,'time':time,'tweets':num_tweets,'followers':num_follow,'retweets':num_retweets,'mentions':num_mentions,'score':rank_score,'hashtags':num_hashtags},
                  columns=['date','time','tweets', 'followers', 'retweets','mentions','score','hashtags'])

df1 = df.groupby(['date', 'time']).agg({'time' : np.max, 'tweets' : np.sum,  'followers' : np.sum, 'retweets' : np.sum, 'mentions' : np.sum, 'score' : np.sum, 'hashtags' : np.sum})

df_5 = pd.DataFrame({'date':date,'time':time_5,'tweets':num_tweets,'followers':num_follow,'retweets':num_retweets,'mentions':num_mentions,'score':rank_score,'hashtags':num_hashtags},
                  columns=['date','time','tweets', 'followers', 'retweets','mentions','score','hashtags'])

df_5 = df_5.groupby(['date', 'time']).agg({'time' : np.max, 'tweets' : np.sum,  'followers' : np.sum, 'retweets' : np.sum, 'mentions' : np.sum, 'score' : np.sum, 'hashtags' : np.sum})

t1 = df1.query('date < 20150201 or (date == 20150201 and time < 8)')
t2 = df_5.query('date == 20150201 and (time >= 800 and time <= 2000)')
t3 = df1.query('date > 20150201 or (date == 20150201 and time > 20)')

X = []
for i in t1.index:
    X.append(t1.loc[i, 'tweets':].values)
X.pop()
X = sm.add_constant(X)
y = t1.loc[t1.index[1]:, 'tweets'].values
model = sm.OLS(y,X).fit()
pred_y = model.predict(X)
print("MSE for %s for before Feb 1: %.6f" % (tweetfile, mean_squared_error(y,pred_y)))
print(model.summary())

X = []
for i in t2.index:
    X.append(t2.loc[i, 'tweets':].values)
X.pop()
X = sm.add_constant(X)
y = t2.loc[t2.index[1]:, 'tweets'].values
model = sm.OLS(y,X).fit()
pred_y = model.predict(X)
print("MSE for %s  for between Feb 1: %.6f" % (tweetfile, mean_squared_error(y,pred_y)))
print(model.summary())

X = []
for i in t3.index:
    X.append(t3.loc[i, 'tweets':].values)
X.pop()
X = sm.add_constant(X)
y = t3.loc[t3.index[1]:, 'tweets'].values
model = sm.OLS(y,X).fit()
pred_y = model.predict(X)
print("MSE for %s  for after Feb 1: %.6f" % (tweetfile, mean_squared_error(y,pred_y)))
print(model.summary())
"""


############################################################################### QUESTION 7 
"""
#hashtags = ['tweets_#gohawks.txt', 'tweets_#gopatriots.txt', 'tweets_#nfl.txt', 'tweets_#patriots.txt', 'tweets_#sb49.txt', 'tweets_#superbowl.txt']
#with open('tweets_#all.txt','w',encoding="utf8") as make:
 #   for file in hashtags:
  #      with open(file,encoding="utf8") as f:
   #         for line in f:
    #            make.write(line)

#print('done making')

tweetfile = 'tweets_#all.txt'
file = []
with open(tweetfile,encoding="utf8") as f:
    for line in f:
        file.append(line)

pst_tz = pytz.timezone('America/Los_Angeles')
date,time,num_tweets,num_follow,num_retweets,num_mentions,rank_score,num_hashtags = [],[],[],[],[],[],[],[]
time_5 = []
for n,line in enumerate(file):
    tw = json.loads(line)
    timestamp = tw['citation_date']
    timestamp = str(datetime.datetime.fromtimestamp(int(timestamp), pst_tz))
    time_split = ''.join(timestamp[0:10].split('-'))
    date.append(int(time_split))
    time.append(int(timestamp[11:13]))

    a,b,c,_ = timestamp.split(':')
    rounded = str(int((int(b)/5))*5).zfill(2)
    time_mins = timestamp[11:13] + rounded
    time_5.append(int(time_mins))
    
    num_tweets.append(1)
    num_follow.append(tw['author']['followers'])
    num_retweets.append(tw['metrics']['citations']['total'])
    num_mentions.append(len(tw['tweet']['entities']['user_mentions']))
    rank_score.append(tw['metrics']['ranking_score'])
    num_hashtags.append(tw['title'].count('#'))
print('done loading')
df = pd.DataFrame({'date':date,'time':time,'tweets':num_tweets,'followers':num_follow,'retweets':num_retweets,'mentions':num_mentions,'score':rank_score,'hashtags':num_hashtags},
                  columns=['date','time','tweets', 'followers', 'retweets','mentions','score','hashtags'])

df_5 = pd.DataFrame({'date':date,'time':time_5,'tweets':num_tweets,'followers':num_follow,'retweets':num_retweets,'mentions':num_mentions,'score':rank_score,'hashtags':num_hashtags},
                  columns=['date','time','tweets', 'followers', 'retweets','mentions','score','hashtags'])

df.to_csv('all.csv')
df_5.to_csv('all_5mins.csv')

df1 = pd.read_csv('all.csv')
df1 = df1.groupby(['date', 'time']).agg({'time' : np.max, 'tweets' : np.sum, 'followers' : np.sum, 'retweets' : np.sum, 'mentions' : np.sum, 'score' : np.sum, 'hashtags' : np.sum})

df_5 = pd.read_csv('all_5mins.csv')
df_5 = df_5.groupby(['date', 'time']).agg({'time' : np.max, 'tweets' : np.sum,  'followers' : np.sum, 'retweets' : np.sum, 'mentions' : np.sum, 'score' : np.sum, 'hashtags' : np.sum})

tweetfile = 'tweets_#all.txt'

t1 = df1.query('date < 20150201 or (date == 20150201 and time < 8)')
t2 = df_5.query('date == 20150201 and (time >= 800 and time <= 2000)')
t3 = df1.query('date > 20150201 or (date == 20150201 and time > 20)')

X = []
for i in t1.index:
    X.append(t1.loc[i, 'tweets':'hashtags'].values)
X.pop()
X = sm.add_constant(X)
y = t1.loc[t1.index[1]:, 'tweets'].values
model = sm.OLS(y,X).fit()
pred_y = model.predict(X)
print("MSE for %s for before Feb 1: %.6f" % (tweetfile, mean_squared_error(y,pred_y)))
print(model.summary())
print('done')

X = []
for i in t2.index:
    X.append(t2.loc[i, 'tweets':'hashtags'].values)
X.pop()
X = sm.add_constant(X)
y = t2.loc[t2.index[1]:, 'tweets'].values
model = sm.OLS(y,X).fit()
pred_y = model.predict(X)
print("MSE for %s  for between Feb 1: %.6f" % (tweetfile, mean_squared_error(y,pred_y)))
print(model.summary())
print('done')

X = []
for i in t3.index:
    X.append(t3.loc[i, 'tweets':'hashtags'].values)
X.pop()
X = sm.add_constant(X)
y = t3.loc[t3.index[1]:, 'tweets'].values
model = sm.OLS(y,X).fit()
pred_y = model.predict(X)
print("MSE for %s  for after Feb 1: %.6f" % (tweetfile, mean_squared_error(y,pred_y)))
print(model.summary())
print('done')
"""

############################################################################### QUESTION 8 
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV, KFold
"""
df = pd.read_csv('all.csv)
df1 = df.groupby(['date', 'time']).agg({'time' : np.max, 'tweets' : np.sum, 'followers' : np.sum, 'retweets' : np.sum,'mentions' : np.sum, 'score' : np.sum, 'hashtags' : np.sum})

X = df1.loc[:,'tweets':].values
y = df1['tweets'].values

param_grid = {'max_depth': [10, 20, 40, 60, 80, 100, 200, None],'max_features': ['auto', 'sqrt'],'min_samples_leaf': [1, 2, 4],'min_samples_split': [2, 5, 10],
              'n_estimators': [200, 400, 600, 800, 1000,1200, 1400, 1600, 1800, 2000]}

rf = RandomForestRegressor()
clf = GridSearchCV(rf,param_grid,cv=KFold(n_splits=5,shuffle=True),scoring='neg_mean_squared_error')
clf.fit(X,y)
print("Best estimator for Random Forest: \n", clf.best_estimator_)
print("Best score for Random Forest: ", -1*clf.best_score_)
print("Best parameters for Random Forest: ", clf.best_params_)
plt.figure()
plt.plot(-1*clf.cv_results_['mean_test_score'])
plt.xlabel("Combination number"); plt.ylabel('Test MSE'); plt.title('Random Forest')
plt.show()

gb = GradientBoostingRegressor()
clf = GridSearchCV(gb,param_grid,cv=KFold(n_splits=5,shuffle=True),scoring='neg_mean_squared_error')
clf.fit(X,y)
print("Best estimator for Gradient Boosting: \n", clf.best_estimator_)
print("Best score for Gradient Boosting: ", -1*clf.best_score_)
print("Best parameters for Gradient Boosting: ", clf.best_params_)
plt.figure()
plt.plot(-1*clf.cv_results_['mean_test_score'])
plt.xlabel("Combination number"); plt.ylabel('Test MSE'); plt.title('Gradient Boosting')
plt.show()
"""


############################################################################### QUESTION 9 

"""
df = pd.read_csv('all.csv')
df1 = df.groupby(['date', 'time']).agg({'time' : np.max, 'tweets' : np.sum, 'followers' : np.sum, 'retweets' : np.sum,'mentions' : np.sum, 'score' : np.sum, 'hashtags' : np.sum})

tweetfile = 'tweets_#all.txt'
X = []
for i in df1.index:
    X.append(df1.loc[i, 'tweets':].values)
X.pop()
X = sm.add_constant(X)
y = df1['tweets'].values[1:]
model = sm.OLS(y,X).fit()
pred_y = model.predict(X)
print("MSE for %s  for : %.6f" % (tweetfile, mean_squared_error(y,pred_y)))
print(model.summary())
"""

############################################################################### QUESTION 10 
"""
df = pd.read_csv('all.csv')
df1 = df.groupby(['date', 'time']).agg({'time' : np.max, 'tweets' : np.sum, 'followers' : np.sum, 'retweets' : np.sum, 'mentions' : np.sum, 'score' : np.sum, 'hashtags' : np.sum})

df_5 = pd.read_csv('all_5mins.csv')
df_5 = df_5.groupby(['date', 'time']).agg({'time' : np.max, 'tweets' : np.sum,  'followers' : np.sum, 'retweets' : np.sum, 'mentions' : np.sum, 'score' : np.sum, 'hashtags' : np.sum})

t1 = df1.query('date < 20150201 or (date == 20150201 and time < 8)')
t2 = df_5.query('date == 20150201 and (time >= 800 and time <= 2000)')
t3 = df1.query('date > 20150201 or (date == 20150201 and time > 20)')

param_grid = {'max_depth': [10, 20, 40, 60, 80, 100, 200, None],'max_features': ['auto', 'sqrt'],'min_samples_leaf': [1, 2, 4],'min_samples_split': [2, 5, 10],
              'n_estimators': [200, 400, 600, 800, 1000,1200, 1400, 1600, 1800, 2000]}

X = t1.loc[:,'tweets':].values
y = t1['tweets'].values

gb = GradientBoostingRegressor()
clf = GridSearchCV(gb,param_grid,cv=KFold(n_splits=5,shuffle=True),scoring='neg_mean_squared_error')
clf.fit(X,y)
print("Best estimator for Gradient Boosting before Feb 1: \n", clf.best_estimator_)
print("Best score for Gradient Boosting before Feb 1: ", -1*clf.best_score_)
print("Best parameters for Gradient Boosting before Feb 1: ", clf.best_params_)
plt.figure()
plt.plot(-1*clf.cv_results_['mean_test_score'])
plt.xlabel("Combination number"); plt.ylabel('Test MSE'); plt.title('Gradient Boosting before Feb 1')
plt.show()

X = t2.loc[:,'tweets':].values
y = t2['tweets'].values

gb = GradientBoostingRegressor()
clf = GridSearchCV(gb,param_grid,cv=KFold(n_splits=5,shuffle=True),scoring='neg_mean_squared_error',verbose=2)
clf.fit(X,y)
print("Best estimator for Gradient Boosting between Feb 1: \n", clf.best_estimator_)
print("Best score for Gradient Boosting between Feb 1: ", -1*clf.best_score_)
print("Best parameters for Gradient Boosting between Feb 1: ", clf.best_params_)
plt.figure()
plt.plot(-1*clf.cv_results_['mean_test_score'])
plt.xlabel("Combination number"); plt.ylabel('Test MSE'); plt.title('Gradient Boosting between Feb 1')
plt.show()

X = t3.loc[:,'tweets':].values
y = t3['tweets'].values

gb = GradientBoostingRegressor()
clf = GridSearchCV(gb,param_grid,cv=KFold(n_splits=5,shuffle=True),scoring='neg_mean_squared_error')
clf.fit(X,y)
print("Best estimator for Gradient Boosting after Feb 1: \n", clf.best_estimator_)
print("Best score for Gradient Boosting after Feb 1: ", -1*clf.best_score_)
print("Best parameters for Gradient Boosting after Feb 1: ", clf.best_params_)
plt.figure()
plt.plot(-1*clf.cv_results_['mean_test_score'])
plt.xlabel("Combination number"); plt.ylabel('Test MSE'); plt.title('Gradient Boosting after Feb 1')
plt.show()
"""

############################################################################### QUESTION 11 
from sklearn.neural_network import MLPRegressor
"""
df = pd.read_csv('all.csv')
df1 = df.groupby(['date', 'time']).agg({'time' : np.max, 'tweets' : np.sum, 'followers' : np.sum, 'retweets' : np.sum, 'mentions' : np.sum, 'score' : np.sum, 'hashtags' : np.sum})

X = df1.loc[:,'tweets':].values
y = df1['tweets'].values

param = {'hidden_layer_sizes':[(100,), (200,200,), (300,300,300,), (400,400,400,400,),
                               (500,500,500,500,500,), (600,600,600,600,600,600,),
                               (700,700,700,700,700,700,700,), (800,800,800,800,800,800,800,800,),
                               (900,900,900,900,900,900,900,900,900,), (1000,1000,1000,1000,1000,1000,1000,1000,1000,1000,),
                               (1100,1100,1100,1100,1100,1100,1100,1100,1100,1100,1100,),
                               (1200,1200,1200,1200,1200,1200,1200,1200,1200,1200,1200,1200,)] }

nn = MLPRegressor()
clf = GridSearchCV(nn,param,cv=KFold(n_splits=5,shuffle=True),scoring='neg_mean_squared_error',verbose=2)
clf.fit(X,y)
print("Best estimator for Neural Network : \n", clf.best_estimator_)
print("Best score for Neural Network : ", -1*clf.best_score_)
print("Best parameters for Neural Network : ", clf.best_params_)
plt.figure()
plt.plot(-1*clf.cv_results_['mean_train_score'])
plt.xlabel("Combination number"); plt.ylabel('Train MSE'); plt.title('Neural Network')
plt.figure()
plt.plot(-1*clf.cv_results_['mean_test_score'])
plt.xlabel("Combination number"); plt.ylabel('Test MSE'); plt.title('Neural Network')
plt.show()
"""

############################################################################### QUESTION 12 
from sklearn.preprocessing import StandardScaler
"""
df = pd.read_csv('all.csv')
df1 = df.groupby(['date', 'time']).agg({'time' : np.max, 'tweets' : np.sum, 'followers' : np.sum, 'retweets' : np.sum, 'mentions' : np.sum, 'score' : np.sum, 'hashtags' : np.sum})

X = df1.loc[:,'tweets':].values
y = df1['tweets'].values

X_ss = StandardScaler().fit_transform(X)

nn = MLPRegressor(hidden_layer_sizes=(1100,1100,1100,1100,1100,1100,1100,1100,1100,1100,1100,))
nn.fit(X_ss,y)
print("Training MSE using StandardScaler with 11 layers and 1100 neurons in each layer: ", nn.loss_)
"""

############################################################################### QUESTION 13 
"""
df = pd.read_csv('all.csv')
df1 = df.groupby(['date', 'time']).agg({'time' : np.max, 'tweets' : np.sum, 'followers' : np.sum, 'retweets' : np.sum, 'mentions' : np.sum, 'score' : np.sum, 'hashtags' : np.sum})

df_5 = pd.read_csv('all_5mins.csv')
df_5 = df_5.groupby(['date', 'time']).agg({'time' : np.max, 'tweets' : np.sum,  'followers' : np.sum, 'retweets' : np.sum, 'mentions' : np.sum, 'score' : np.sum, 'hashtags' : np.sum})

t1 = df1.query('date < 20150201 or (date == 20150201 and time < 8)')
t2 = df_5.query('date == 20150201 and (time >= 800 and time <= 2000)')
t3 = df1.query('date > 20150201 or (date == 20150201 and time > 20)')

param = {'hidden_layer_sizes':[(100,), (200,200,), (300,300,300,), (400,400,400,400,),
                               (500,500,500,500,500,), (600,600,600,600,600,600,),
                               (700,700,700,700,700,700,700,), (800,800,800,800,800,800,800,800,),
                               (900,900,900,900,900,900,900,900,900,), (1000,1000,1000,1000,1000,1000,1000,1000,1000,1000,),
                               (1100,1100,1100,1100,1100,1100,1100,1100,1100,1100,1100,),
                               (1200,1200,1200,1200,1200,1200,1200,1200,1200,1200,1200,1200,)] }

X = t1.loc[:,'tweets':].values
X = StandardScaler().fit_transform(X)
y = t1['tweets'].values

nn = MLPRegressor()
clf = GridSearchCV(nn,param,cv=KFold(n_splits=5,shuffle=True),scoring='neg_mean_squared_error',verbose=2)
clf.fit(X,y)
print("Best estimator for Neural Network Before Feb 1: \n", clf.best_estimator_)
print("Best score for Neural Network Before Feb 1: ", -1*clf.best_score_)
print("Best parameters for Neural Network Before Feb 1: ", clf.best_params_)
plt.figure()
plt.plot(-1*clf.cv_results_['mean_test_score'])
plt.xlabel("Combination number"); plt.ylabel('Test MSE'); plt.title('Neural Network Before Feb 1')
plt.show()

X = t2.loc[:,'tweets':].values
X = StandardScaler().fit_transform(X)
y = t2['tweets'].values

nn = MLPRegressor()
clf = GridSearchCV(nn,param,cv=KFold(n_splits=5,shuffle=True),scoring='neg_mean_squared_error',verbose=2)
clf.fit(X,y)
print("Best estimator for Neural Network Between Feb 1: \n", clf.best_estimator_)
print("Best score for Neural Network Between Feb 1: ", -1*clf.best_score_)
print("Best parameters for Neural Network Between Feb 1: ", clf.best_params_)
plt.figure()
plt.plot(-1*clf.cv_results_['mean_test_score'])
plt.xlabel("Combination number"); plt.ylabel('Test MSE'); plt.title('Neural Network Between Feb 1')
plt.show()

X = t3.loc[:,'tweets':].values
X = StandardScaler().fit_transform(X)
y = t3['tweets'].values

nn = MLPRegressor()
clf = GridSearchCV(nn,param,cv=KFold(n_splits=5,shuffle=True),scoring='neg_mean_squared_error',verbose=2)
clf.fit(X,y)
print("Best estimator for Neural Network After Feb 1: \n", clf.best_estimator_)
print("Best score for Neural Network After Feb 1: ", -1*clf.best_score_)
print("Best parameters for Neural Network After Feb 1: ", clf.best_params_)
plt.figure()
plt.plot(-1*clf.cv_results_['mean_test_score'])
plt.xlabel("Combination number"); plt.ylabel('Test MSE'); plt.title('Neural Network After Feb 1')
plt.show()
"""

############################################################################### QUESTION 14 
"""
df = pd.read_csv('all.csv')
df1 = df.groupby(['date', 'time']).agg({'time' : np.max, 'tweets' : np.sum, 'followers' : np.sum, 'retweets' : np.sum, 'mentions' : np.sum, 'score' : np.sum, 'hashtags' : np.sum})

df_5 = pd.read_csv('all_5mins.csv')
df_5 = df_5.groupby(['date', 'time']).agg({'time' : np.max, 'tweets' : np.sum,  'followers' : np.sum, 'retweets' : np.sum, 'mentions' : np.sum, 'score' : np.sum, 'hashtags' : np.sum})

t1 = df1.query('date < 20150201 or (date == 20150201 and time < 8)')
t2 = df_5.query('date == 20150201 and (time >= 800 and time <= 2000)')
t3 = df1.query('date > 20150201 or (date == 20150201 and time > 20)')


X = t1.loc[:,'tweets':].values
ss = StandardScaler()
X = ss.fit_transform(X)
y = t1['tweets'].values
nn = MLPRegressor(hidden_layer_sizes=(600,600,600,600,600,600,))
nn.fit(X,y)
tweetfile = 'sample2_period1.txt'
file = []
with open(tweetfile,encoding="utf8") as f:
    for line in f:
        file.append(line)
pst_tz = pytz.timezone('America/Los_Angeles')
date,time,num_tweets,num_follow,num_retweets,num_mentions,rank_score,num_hashtags = [],[],[],[],[],[],[],[]
time_5 = []
for n,line in enumerate(file):
    tw = json.loads(line)
    timestamp = tw['citation_date']
    timestamp = str(datetime.datetime.fromtimestamp(int(timestamp), pst_tz))
    time_split = ''.join(timestamp[0:10].split('-'))
    date.append(int(time_split))
    time.append(int(timestamp[11:13]))
    a,b,c,_ = timestamp.split(':')
    rounded = str(int((int(b)/5))*5).zfill(2)
    time_mins = timestamp[11:13] + rounded
    time_5.append(int(time_mins))
    num_tweets.append(1)
    num_follow.append(tw['author']['followers'])
    num_retweets.append(tw['metrics']['citations']['total'])
    num_mentions.append(len(tw['tweet']['entities']['user_mentions']))
    rank_score.append(tw['metrics']['ranking_score'])
    num_hashtags.append(tw['title'].count('#'))
sp = pd.DataFrame({'date':date,'time':time,'tweets':num_tweets,'followers':num_follow,'retweets':num_retweets,'mentions':num_mentions,'score':rank_score,'hashtags':num_hashtags},
                  columns=['date','time','tweets', 'followers', 'retweets','mentions','score','hashtags'])
sp = sp.groupby(['date', 'time']).agg({'time' : np.max, 'tweets' : np.sum,  'followers' : np.sum, 'retweets' : np.sum, 'mentions' : np.sum, 'score' : np.sum, 'hashtags' : np.sum})
X_test = sp.loc[:,'tweets':].values
X_test = ss.transform(X_test)
y_test = sp['tweets'].values
y_pred = nn.predict(X_test)
print("Test MSE for %s using Neural Network: %.6f" % (tweetfile, mean_squared_error(y_test,y_pred)))
plt.figure()
plt.plot(y_test,y_pred,'ro')
plt.xlabel('Actual number of tweets');plt.ylabel('Predicted number of tweets'); plt.title('Before Feb 1')
plt.show()


X = t2.loc[:,'tweets':].values
ss = StandardScaler()
X = ss.fit_transform(X)
y = t2['tweets'].values
nn = MLPRegressor(hidden_layer_sizes=(500,500,500,500,500,))
nn.fit(X,y)
tweetfile = 'sample2_period2.txt'
file = []
with open(tweetfile,encoding="utf8") as f:
    for line in f:
        file.append(line)
pst_tz = pytz.timezone('America/Los_Angeles')
date,time,num_tweets,num_follow,num_retweets,num_mentions,rank_score,num_hashtags = [],[],[],[],[],[],[],[]
time_5 = []
for n,line in enumerate(file):
    tw = json.loads(line)
    timestamp = tw['citation_date']
    timestamp = str(datetime.datetime.fromtimestamp(int(timestamp), pst_tz))
    time_split = ''.join(timestamp[0:10].split('-'))
    date.append(int(time_split))
    time.append(int(timestamp[11:13]))
    a,b,c,_ = timestamp.split(':')
    rounded = str(int((int(b)/5))*5).zfill(2)
    time_mins = timestamp[11:13] + rounded
    time_5.append(int(time_mins))
    num_tweets.append(1)
    num_follow.append(tw['author']['followers'])
    num_retweets.append(tw['metrics']['citations']['total'])
    num_mentions.append(len(tw['tweet']['entities']['user_mentions']))
    rank_score.append(tw['metrics']['ranking_score'])
    num_hashtags.append(tw['title'].count('#'))
sp = pd.DataFrame({'date':date,'time':time_5,'tweets':num_tweets,'followers':num_follow,'retweets':num_retweets,'mentions':num_mentions,'score':rank_score,'hashtags':num_hashtags},
                  columns=['date','time','tweets', 'followers', 'retweets','mentions','score','hashtags'])
sp = sp.groupby(['date', 'time']).agg({'time' : np.max, 'tweets' : np.sum,  'followers' : np.sum, 'retweets' : np.sum, 'mentions' : np.sum, 'score' : np.sum, 'hashtags' : np.sum})
X_test = sp.loc[:,'tweets':].values
X_test = ss.transform(X_test)
y_test = sp['tweets'].values
y_pred = nn.predict(X_test)
print("Test MSE for %s using Neural Network: %.6f" % (tweetfile, mean_squared_error(y_test,y_pred)))
plt.figure()
plt.plot(y_test,y_pred,'ro')
plt.xlabel('Actual number of tweets');plt.ylabel('Predicted number of tweets'); plt.title('Between Feb 1')
plt.show()


X = t3.loc[:,'tweets':].values
ss = StandardScaler()
X = ss.fit_transform(X)
y = t3['tweets'].values
nn = MLPRegressor(hidden_layer_sizes=(800,800,800,800,800,800,800,800,))
nn.fit(X,y)
tweetfile = 'sample2_period3.txt'
file = []
with open(tweetfile,encoding="utf8") as f:
    for line in f:
        file.append(line)
pst_tz = pytz.timezone('America/Los_Angeles')
date,time,num_tweets,num_follow,num_retweets,num_mentions,rank_score,num_hashtags = [],[],[],[],[],[],[],[]
time_5 = []
for n,line in enumerate(file):
    tw = json.loads(line)
    timestamp = tw['citation_date']
    timestamp = str(datetime.datetime.fromtimestamp(int(timestamp), pst_tz))
    time_split = ''.join(timestamp[0:10].split('-'))
    date.append(int(time_split))
    time.append(int(timestamp[11:13]))
    a,b,c,_ = timestamp.split(':')
    rounded = str(int((int(b)/5))*5).zfill(2)
    time_mins = timestamp[11:13] + rounded
    time_5.append(int(time_mins))
    num_tweets.append(1)
    num_follow.append(tw['author']['followers'])
    num_retweets.append(tw['metrics']['citations']['total'])
    num_mentions.append(len(tw['tweet']['entities']['user_mentions']))
    rank_score.append(tw['metrics']['ranking_score'])
    num_hashtags.append(tw['title'].count('#'))
sp = pd.DataFrame({'date':date,'time':time,'tweets':num_tweets,'followers':num_follow,'retweets':num_retweets,'mentions':num_mentions,'score':rank_score,'hashtags':num_hashtags},
                  columns=['date','time','tweets', 'followers', 'retweets','mentions','score','hashtags'])
sp = sp.groupby(['date', 'time']).agg({'time' : np.max, 'tweets' : np.sum,  'followers' : np.sum, 'retweets' : np.sum, 'mentions' : np.sum, 'score' : np.sum, 'hashtags' : np.sum})
X_test = sp.loc[:,'tweets':].values
X_test = ss.transform(X_test)
y_test = sp['tweets'].values
y_pred = nn.predict(X_test)
print("Test MSE for %s using Neural Network: %.6f" % (tweetfile, mean_squared_error(y_test,y_pred)))
plt.figure()
plt.plot(y_test,y_pred,'ro')
plt.xlabel('Actual number of tweets');plt.ylabel('Predicted number of tweets'); plt.title('After Feb 1')
plt.show()
"""

############################################################################### QUESTION 15 
import re
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score, roc_curve, auc
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC
import itertools

"""
tweetfile = 'tweets_#superbowl.txt'
file = []
with open(tweetfile,encoding="utf8") as f:
    for line in f:
        file.append(line)

title, city = [],[]
for n,line in enumerate(file): # 1 for Mass, 0 for Wash
    tw = json.loads(line)
    loc = tw['tweet']['user']['location']
    if (re.match('.*MA.*', loc) or re.match('.*Mass.*', loc)):
        title.append(tw['title'])
        city.append(1)
    elif ((re.match('.*WA.*', loc) or re.match('.*Wash.*', loc)) and not re.match('.*DC.*', loc) and not re.match('.*D\\.C\\..*', loc)):
        title.append(tw['title'])
        city.append(0)

def penn2morphy(penntag):
    morphy_tag = {'NN':'n', 'JJ':'a','VB':'v', 'RB':'r'}
    try:
        return morphy_tag[penntag[:2]]
    except:
        return 'n'

lemmer = WordNetLemmatizer()
def lemmatize_sent(text):
    return [lemmer.lemmatize(word.lower(), pos=penn2morphy(tag)) for word, tag in pos_tag(nltk.word_tokenize(text))]

for n,sent in enumerate(title):
    sent = lemmatize_sent(sent) # lemmatize
    sent = [i for i in sent if i not in string.punctuation] # remove punctuation
    sent = [i for i in sent if not i.isdigit()] # remove numbers
    title[n] = " ".join(sent)

tfidf_vect = TfidfVectorizer(stop_words='english',min_df=3)
X = tfidf_vect.fit_transform(title) # making the tfidf train matrix
LSI = TruncatedSVD(n_components=50, random_state=42)
X_lsi = LSI.fit_transform(X) # performing LSI on the tfidf train matrix

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),horizontalalignment="center",color="white" if cm[i, j] > thresh else "black")
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

def print_classifier_metrics(y_test,y_pred,name="",average='binary'):
    print("Accuracy score for %s: %f" %(name,accuracy_score(y_test,y_pred)))
    print("Recall score for %s: %f" % (name,recall_score(y_test,y_pred,average=average)))
    print("Precision score for %s: %f" % (name,precision_score(y_test,y_pred,average=average)))
    print("F-1 score for %s: %f" % (name,f1_score(y_test,y_pred,average=average)))

def plot_roc_curve(y_test,decision_function,name=""):
    fpr = dict();tpr = dict();roc_auc = dict()
    fpr, tpr, thresholds = roc_curve(y_test, decision_function)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',lw=lw, label='ROC curve (area = %0.4f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0]); plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate');
    plt.title('%s ROC curve' % name);plt.legend(loc="lower right")


classifier = 'Gaussian Naive Bayes'
lr = GaussianNB() 
clf = GridSearchCV(lr,{},cv=5,scoring='accuracy')
y_pred = clf.fit(X_lsi,city).best_estimator_.predict(X_lsi)
print_classifier_metrics(city,y_pred,name=classifier)
cv_cm = confusion_matrix(city,y_pred)
class_names = ['Massachusetts','Washington']
plt.figure(); plot_confusion_matrix(cv_cm, classes=class_names, title=classifier); plt.show()
plt.figure(); plot_roc_curve(city,clf.best_estimator_.predict_proba(X_lsi)[:,1],name=classifier); plt.show()
"""


############################################################################### QUESTION 16 
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
"""
sid = SentimentIntensityAnalyzer()
tweetfile = 'tweets_#gopatriots.txt'
file = []
with open(tweetfile,encoding="utf8") as f:
    for line in f:
        file.append(line)
pst_tz = pytz.timezone('America/Los_Angeles')
date,time,num_tweets,num_follow,num_retweets,num_mentions,rank_score,num_hashtags = [],[],[],[],[],[],[],[]
time_5 = []
pos,neu,neg,imp = [],[],[],[]
for n,line in enumerate(file):
    tw = json.loads(line)
    timestamp = tw['citation_date']
    timestamp = str(datetime.datetime.fromtimestamp(int(timestamp), pst_tz))
    time_split = ''.join(timestamp[0:10].split('-'))
    date.append(int(time_split))
    time.append(int(timestamp[11:13]))
    title = tw['title']
    pos.append(sid.polarity_scores(title)['compound'])
    #neu.append(sid.polarity_scores(title)['neu'])
    #neg.append(sid.polarity_scores(title)['neg'])
    imp.append(tw['metrics']['impressions'])
    a,b,c,_ = timestamp.split(':')
    rounded = str(int((int(b)/5))*5).zfill(2)
    time_mins = timestamp[11:13] + rounded
    time_5.append(int(time_mins))
    num_tweets.append(1)
    num_follow.append(tw['author']['followers'])
    num_retweets.append(tw['metrics']['citations']['total'])
    num_mentions.append(len(tw['tweet']['entities']['user_mentions']))
    rank_score.append(tw['metrics']['ranking_score'])
    num_hashtags.append(tw['title'].count('#'))
sp = pd.DataFrame({'date':date,'time':time_5,'tweets':num_tweets,'followers':num_follow,'retweets':num_retweets,'imp':imp,'pos':pos},
                  columns=['date','time','tweets', 'followers', 'retweets','imp','pos'])
sp = sp.groupby(['date', 'time']).agg({'time' : np.max, 'tweets' : np.sum,  'followers' : np.sum, 'retweets' : np.sum,'imp':np.sum,'pos':np.sum})

t2 = sp.query('date == 20150201 and (time >= 1500 and time <= 2100)')
plt.figure()
plt.plot(t2['imp'].values)
plt.xlabel('Number of 5-min window');plt.ylabel('Sum of impressions'); 
plt.show()

plt.figure()
plt.plot(t2['pos'].values,'r')
plt.xlabel('Number of 5-min window');plt.ylabel('Sum of Compound score'); 
plt.show()



def clean(title):
    title = nltk.word_tokenize(title)
    title = [i.lower() for i in title if i not in stopwords.words('english')]
    for n,i in enumerate(title):
        try:            
            if i == '#':
                title[n] = ""
                title[n+1] = ""
            elif i == 'https':
                title[n] = ""
                title[n+1] = ""
                title[n+2] = ""
        except IndexError:
            title[n] = ""
    title = " ".join(title)
    return title

sid = SentimentIntensityAnalyzer()
tweetfile = 'tweets_#nfl.txt'
file = []
with open(tweetfile,encoding="utf8") as f:
    for line in f:
        file.append(line)
pst_tz = pytz.timezone('America/Los_Angeles')
date,time,num_tweets,num_follow,num_retweets,num_mentions,rank_score,num_hashtags = [],[],[],[],[],[],[],[]
time_5 = []
pos,neu,neg,imp = [],[],[],[]
for n,line in enumerate(file):
    tw = json.loads(line)
    timestamp = tw['citation_date']
    timestamp = str(datetime.datetime.fromtimestamp(int(timestamp), pst_tz))
    time_split = ''.join(timestamp[0:10].split('-'))
    date.append(int(time_split))
    time.append(int(timestamp[11:13]))
    title = tw['title']
    pos.append(sid.polarity_scores(title)['compound'])
    #neu.append(sid.polarity_scores(title)['neu'])
    #neg.append(sid.polarity_scores(title)['neg'])
    imp.append(tw['metrics']['impressions'])
    a,b,c,_ = timestamp.split(':')
    rounded = str(int((int(b)/5))*5).zfill(2)
    time_mins = timestamp[11:13] + rounded
    time_5.append(int(time_mins))
    num_tweets.append(1)
    num_follow.append(tw['author']['followers'])
    num_retweets.append(tw['metrics']['citations']['total'])
    num_mentions.append(len(tw['tweet']['entities']['user_mentions']))
    rank_score.append(tw['metrics']['ranking_score'])
    num_hashtags.append(tw['title'].count('#'))
sp = pd.DataFrame({'date':date,'time':time_5,'tweets':num_tweets,'followers':num_follow,'retweets':num_retweets,'imp':imp,'pos':pos},
                  columns=['date','time','tweets', 'followers', 'retweets','imp','pos'])
sp = sp.groupby(['date', 'time']).agg({'time' : np.max, 'tweets' : np.sum,  'followers' : np.sum, 'retweets' : np.sum,'imp':np.sum,'pos':np.sum})

t2 = sp.query('date == 20150201 and (time >= 1500 and time <= 2100)')

X = t2.loc[:,'tweets':'imp'].values
ss = StandardScaler()
X = ss.fit_transform(X)
y = t2['pos'].values
param = {'hidden_layer_sizes':[(100,), (200,200,), (300,300,300,), (400,400,400,400,),
                               (500,500,500,500,500,), (600,600,600,600,600,600,),
                               (700,700,700,700,700,700,700,), (800,800,800,800,800,800,800,800,),
                               (900,900,900,900,900,900,900,900,900,), (1000,1000,1000,1000,1000,1000,1000,1000,1000,1000,),
                               (1100,1100,1100,1100,1100,1100,1100,1100,1100,1100,1100,),
                               (1200,1200,1200,1200,1200,1200,1200,1200,1200,1200,1200,1200,)] }

nn = MLPRegressor()
clf = GridSearchCV(nn,param,cv=KFold(n_splits=5,shuffle=True),scoring='neg_mean_squared_error',verbose=2)
clf.fit(X,y)
print("Best estimator for Neural Network: \n", clf.best_estimator_)
print("Best score for Neural Network: ", -1*clf.best_score_)
print("Best parameters for Neural Network: ", clf.best_params_)


sid = SentimentIntensityAnalyzer()
tweetfile = 'tweets_#gopatriots.txt'
file = []
with open(tweetfile,encoding="utf8") as f:
    for line in f:
        file.append(line)
pst_tz = pytz.timezone('America/Los_Angeles')
date,time,num_tweets,num_follow,num_retweets,num_mentions,rank_score,num_hashtags = [],[],[],[],[],[],[],[]
time_5 = []
pos,neu,neg,imp = [],[],[],[]
for n,line in enumerate(file):
    tw = json.loads(line)
    timestamp = tw['citation_date']
    timestamp = str(datetime.datetime.fromtimestamp(int(timestamp), pst_tz))
    time_split = ''.join(timestamp[0:10].split('-'))
    date.append(int(time_split))
    time.append(int(timestamp[11:13]))
    title = tw['title']
    pos.append(sid.polarity_scores(title)['compound'])
    #neu.append(sid.polarity_scores(title)['neu'])
    #neg.append(sid.polarity_scores(title)['neg'])
    imp.append(tw['metrics']['impressions'])
    a,b,c,_ = timestamp.split(':')
    rounded = str(int((int(b)/5))*5).zfill(2)
    time_mins = timestamp[11:13] + rounded
    time_5.append(int(time_mins))
    num_tweets.append(1)
    num_follow.append(tw['author']['followers'])
    num_retweets.append(tw['metrics']['citations']['total'])
    num_mentions.append(len(tw['tweet']['entities']['user_mentions']))
    rank_score.append(tw['metrics']['ranking_score'])
    num_hashtags.append(tw['title'].count('#'))
sp = pd.DataFrame({'date':date,'time':time_5,'tweets':num_tweets,'followers':num_follow,'retweets':num_retweets,'imp':imp,'pos':pos},
                  columns=['date','time','tweets', 'followers', 'retweets','imp','pos'])
sp = sp.groupby(['date', 'time']).agg({'time' : np.max, 'tweets' : np.sum,  'followers' : np.sum, 'retweets' : np.sum,'imp':np.sum,'pos':np.sum})

sp = sp.query('date == 20150201 and (time >= 1500 and time <= 2100)')

X_test = sp.loc[:,'tweets':'imp'].values
X_test = ss.transform(X_test)
y_test = sp['pos'].values
y_pred = clf.best_estimator_.predict(X_test)
print("Test MSE for %s using Neural Network: %.6f" % (tweetfile, mean_squared_error(y_test,y_pred)))
plt.figure()
plt.tight_layout()
plt.plot(y_test,y_pred,'ro')
plt.xlabel('Actual sum of Compound');plt.ylabel('Predicted sum of Compound'); plt.title(tweetfile)
plt.show()


sid = SentimentIntensityAnalyzer()
tweetfile = 'tweets_#gohawks.txt'
file = []
with open(tweetfile,encoding="utf8") as f:
    for line in f:
        file.append(line)
pst_tz = pytz.timezone('America/Los_Angeles')
date,time,num_tweets,num_follow,num_retweets,num_mentions,rank_score,num_hashtags = [],[],[],[],[],[],[],[]
time_5 = []
pos,neu,neg,imp = [],[],[],[]
for n,line in enumerate(file):
    tw = json.loads(line)
    timestamp = tw['citation_date']
    timestamp = str(datetime.datetime.fromtimestamp(int(timestamp), pst_tz))
    time_split = ''.join(timestamp[0:10].split('-'))
    date.append(int(time_split))
    time.append(int(timestamp[11:13]))
    title = tw['title']
    pos.append(sid.polarity_scores(title)['compound'])
    #neu.append(sid.polarity_scores(title)['neu'])
    #neg.append(sid.polarity_scores(title)['neg'])
    imp.append(tw['metrics']['impressions'])
    a,b,c,_ = timestamp.split(':')
    rounded = str(int((int(b)/5))*5).zfill(2)
    time_mins = timestamp[11:13] + rounded
    time_5.append(int(time_mins))
    num_tweets.append(1)
    num_follow.append(tw['author']['followers'])
    num_retweets.append(tw['metrics']['citations']['total'])
    num_mentions.append(len(tw['tweet']['entities']['user_mentions']))
    rank_score.append(tw['metrics']['ranking_score'])
    num_hashtags.append(tw['title'].count('#'))
sp = pd.DataFrame({'date':date,'time':time_5,'tweets':num_tweets,'followers':num_follow,'retweets':num_retweets,'imp':imp,'pos':pos},
                  columns=['date','time','tweets', 'followers', 'retweets','imp','pos'])
sp = sp.groupby(['date', 'time']).agg({'time' : np.max, 'tweets' : np.sum,  'followers' : np.sum, 'retweets' : np.sum,'imp':np.sum,'pos':np.sum})

sp = sp.query('date == 20150201 and (time >= 1500 and time <= 2100)')

X_test = sp.loc[:,'tweets':'imp'].values
X_test = ss.transform(X_test)
y_test = sp['pos'].values
y_pred = clf.best_estimator_.predict(X_test)
print("Test MSE for %s using Neural Network: %.6f" % (tweetfile, mean_squared_error(y_test,y_pred)))
plt.figure()
plt.tight_layout()
plt.plot(y_test,y_pred,'ro')
plt.xlabel('Actual sum of Compound');plt.ylabel('Predicted sum of Compound'); plt.title(tweetfile)
plt.show()
"""


