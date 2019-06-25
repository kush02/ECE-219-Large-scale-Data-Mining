"""
    Kushagra Rastogi
    304640248
    ECE 219
    Project 1
"""

import sys
import numpy as np
np.random.seed(42)

import matplotlib.pyplot as plt
import pandas as pd

import random
random.seed(42)

import nltk, string

######################################################## QUESTION 1
from sklearn.datasets import fetch_20newsgroups

newsgroups = fetch_20newsgroups(subset = 'train',shuffle = True, random_state = 42) # getting training data
titles = []; docs = [];
for cat in newsgroups.target_names:
    titles.append(cat)
    temp = fetch_20newsgroups(subset = 'train',categories = [cat], shuffle = True, random_state = 42) # getting category data
    docs.append(len(temp.data))

plt.figure()
plt.bar(np.arange(1,21), docs,tick_label=titles) # bar plot 
plt.xticks(np.arange(1,21),titles,rotation='vertical')
plt.xlabel("Categories"); plt.ylabel("Number of training documents"); plt.title("Number of documents in each category")
plt.tight_layout()


######################################################## QUESTION 2
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
from sklearn.feature_extraction.text import TfidfVectorizer

def penn2morphy(penntag):
    morphy_tag = {'NN':'n', 'JJ':'a','VB':'v', 'RB':'r'}
    try:
        return morphy_tag[penntag[:2]]
    except:
        return 'n'

lemmer = WordNetLemmatizer()
def lemmatize_sent(text):
    return [lemmer.lemmatize(word.lower(), pos=penn2morphy(tag)) for word, tag in pos_tag(nltk.word_tokenize(text))]

categories = ['comp.graphics', 'comp.os.ms-windows.misc','comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware','rec.autos', 'rec.motorcycles','rec.sport.baseball', 'rec.sport.hockey']

train_dataset = fetch_20newsgroups(subset = 'train', categories = categories, shuffle = True, random_state = 42)# getting training data
for n,sent in enumerate(train_dataset.data):
    sent = lemmatize_sent(sent) # lemmatize
    sent = [i for i in sent if i not in string.punctuation] # remove punctuation
    sent = [i for i in sent if not i.isdigit()] # remove numbers
    train_dataset.data[n] = " ".join(sent)

test_dataset = fetch_20newsgroups(subset = 'test', categories = categories, shuffle = True, random_state = 42) # getting test data
for n,sent in enumerate(test_dataset.data):
    sent = lemmatize_sent(sent) # lemmatize
    sent = [i for i in sent if i not in string.punctuation] # remove punctuation
    sent = [i for i in sent if not i.isdigit()] # remove numbers
    test_dataset.data[n] = " ".join(sent)

tfidf_vect = TfidfVectorizer(stop_words='english',min_df=3)

X_train_tfidf = tfidf_vect.fit_transform(train_dataset.data) # making the tfidf train matrix
print("Shape of TF-IDF train matrix: ", X_train_tfidf.shape)

X_test_tfidf = tfidf_vect.transform(test_dataset.data) # transforming the test data into the tfidf test matrix
print("Shape of TF-IDF test matrix: ", X_test_tfidf.shape)


######################################################### QUESTION 3
from sklearn.decomposition import TruncatedSVD
from sklearn.utils.extmath import randomized_svd
from sklearn.decomposition import NMF

LSI = TruncatedSVD(n_components=50, random_state=42)
X_train_LSI = LSI.fit_transform(X_train_tfidf) # performing LSI on the tfidf train matrix
X_test_LSI = LSI.transform(X_test_tfidf) # performing LSI on the tfidf test matrix
U,Sig,VT = randomized_svd(X_train_tfidf,n_components=50,random_state=42) # obtaining the left singular matrix, singular values and right singular matrix
Sig = np.diag(Sig)
print("Error for LSI: ", np.sum(np.array(X_train_tfidf - U.dot(Sig).dot(VT))**2))


nmf = NMF(n_components=50, init='random', random_state=42)
X_train_NMF = nmf.fit_transform(X_train_tfidf) # performing NMF on the tfidf train matrix
X_test_NMF = nmf.transform(X_test_tfidf) # performing NMF on the tfidf test matrix
H = nmf.components_
print("Error for NMF: ", np.sum(np.array(X_train_tfidf - X_train_NMF.dot(H))**2))


########################################################## QUESTION 4
from sklearn.svm import LinearSVC
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score, roc_curve, auc
import itertools
from sklearn.model_selection import GridSearchCV

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

hardSVM = LinearSVC(C=1000,random_state=42)
softSVM = LinearSVC(C=0.0001,random_state=42)

y_train = [int(i/4) for i in train_dataset.target] # making the train labels compatible for binary classification
y_test = [int(i/4) for i in test_dataset.target] # making the test labels compatible for binary classification

y_pred_hardSVM = hardSVM.fit(X_train_LSI,y_train).predict(X_test_LSI) # predicting labels for hard margin SVM
y_pred_softSVM = softSVM.fit(X_train_LSI,y_train).predict(X_test_LSI) # predicting labels for soft margin SVM

print_classifier_metrics(y_test,y_pred_hardSVM,name="Hard Margin SVM")
print_classifier_metrics(y_test,y_pred_softSVM,name="Soft Margin SVM")

class_names = ['Computer Technology', 'Recreation Activity']
hardSVM_cm = confusion_matrix(y_test,y_pred_hardSVM) # Hard SVM Confusion matrix
plt.figure(); plot_confusion_matrix(hardSVM_cm, classes=class_names, title='Hard SVM Confusion Matrix') 
softSVM_cm = confusion_matrix(y_test,y_pred_softSVM) # Soft SVM Confusion matrix
plt.figure(); plot_confusion_matrix(softSVM_cm, classes=class_names, title='Soft SVM Confusion Matrix')

plot_roc_curve(y_test,hardSVM.decision_function(X_test_LSI),name="Hard Margin SVM") # Hard SVM ROC curve 
plot_roc_curve(y_test,softSVM.decision_function(X_test_LSI),name="Soft Margin SVM") # Soft SVM ROC curve 

svc = LinearSVC(random_state=42) 
params = {'C':[0.001,0.01,0.1,1,10,100,1000]}
clf = GridSearchCV(svc,params,cv=5,scoring='accuracy') # Finding the best gamma using cross validation
clf.fit(X_train_LSI,y_train)

x = [0.001,0.01,0.1,1,10,100,1000]
y = clf.cv_results_['mean_test_score']
fig = plt.figure()
ax = fig.add_subplot(111)
plt.scatter(x,y)
for xy in zip(x, y):                                       
    ax.annotate('(%s, %.5f)' % xy, xy=xy, textcoords='data')
plt.xlabel('C'); plt.ylabel('Mean Test Score'); plt.title("Linear SVM")
plt.grid()

y_pred_cv = clf.best_estimator_.predict(X_test_LSI)
best_svm_gamma = clf.best_estimator_.C

print("Grid search results for SVM: ", clf.cv_results_)
print("Best estimator for SVM: ", clf.best_estimator_)
print("Best parameters for SVM: ", clf.best_params_)
print("Best score for SVM: ", clf.best_score_)
print("Best Gamma for SVM: ", best_svm_gamma)
print_classifier_metrics(y_test,y_pred_cv,name="Best Gamma SVM")

cv_cm = confusion_matrix(y_test,y_pred_cv) # Best Gamma SVM confusion matrix
plt.figure(); plot_confusion_matrix(cv_cm, classes=class_names, title='Best Gamma SVM Confusion Matrix')

plot_roc_curve(y_test,clf.best_estimator_.decision_function(X_test_LSI),name="Best Gamma SVM") # Best Gamma SVM ROC curve 


########################################################## QUESTION 5
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(C=10**10,random_state=42) # Logistic regression without regularization
y_pred_lr = lr.fit(X_train_LSI,y_train).predict(X_test_LSI)
print("Coefficients learned by logistic regression without regularization: ", lr.coef_)
print_classifier_metrics(y_test,y_pred_lr,name="Logistic Regression without regularization")
lr_cm = confusion_matrix(y_test,y_pred_lr) # logistic regression without regularization confusion matrix
plt.figure(); plot_confusion_matrix(lr_cm, classes=class_names, title='Logistic Regression Confusion Matrix')
plot_roc_curve(y_test,lr.decision_function(X_test_LSI),name="Logistic Regression") # logistic regression without regularization roc curve

lr_l2 = LogisticRegression(random_state=42) # logistic regression with L2 regularization
clf_l2 = GridSearchCV(lr_l2,params,cv=5,scoring='accuracy') # grid search for best gamma for L2 regularization
y_pred_l2 = clf_l2.fit(X_train_LSI,y_train).best_estimator_.predict(X_test_LSI)
best_l2_gamma = clf_l2.best_estimator_.C

x = [0.001,0.01,0.1,1,10,100,1000]
y = clf_l2.cv_results_['mean_test_score']
fig = plt.figure()
ax = fig.add_subplot(111)
plt.scatter(x,y)
for xy in zip(x, y):                                       
    ax.annotate('(%s, %.5f)' % xy, xy=xy, textcoords='data')
plt.xlabel('C'); plt.ylabel('Mean Test Score'); plt.title("Logistic Regression with L2 regularization")
plt.grid()


print("Grid search results for logistic regression with l-2 regularization: ", clf_l2.cv_results_)
print("Best estimator for logistic regression with l-2 regularization: ", clf_l2.best_estimator_)
print("Best parameters for logistic regression with l-2 regularization: ", clf_l2.best_params_)
print("Best score for logistic regression with l-2 regularization: ", clf_l2.best_score_)
print("Best Gamma for logistic regression with l-2 regularization: ", best_l2_gamma)
print("Coefficients learned by logistic regression with l-2 regularization: ", clf_l2.best_estimator_.coef_)
print_classifier_metrics(y_test,y_pred_l2,name="Logistic Regression with l-2 regularization")


lr_l1 = LogisticRegression(penalty='l1',random_state=42) # logistic regression with L1 regularization
clf_l1 = GridSearchCV(lr_l1,params,cv=5,scoring='accuracy') # grid search for best gamma for L1 regularization
y_pred_l1 = clf_l1.fit(X_train_LSI,y_train).best_estimator_.predict(X_test_LSI)
best_l1_gamma = clf_l1.best_estimator_.C

x = [0.001,0.01,0.1,1,10,100,1000]
y = clf_l1.cv_results_['mean_test_score']
fig = plt.figure()
ax = fig.add_subplot(111)
plt.scatter(x,y)
for xy in zip(x, y):                                       
    ax.annotate('(%s, %.5f)' % xy, xy=xy, textcoords='data')
plt.xlabel('C'); plt.ylabel('Mean Test Score'); plt.title("Logistic Regression with L1 regularization")
plt.grid()

print("Grid search results for logistic regression with l-1 regularization: ", clf_l1.cv_results_)
print("Best estimator for logistic regression with l-1 regularization: ", clf_l1.best_estimator_)
print("Best parameters for logistic regression with l-1 regularization: ", clf_l1.best_params_)
print("Best score for logistic regression with l-1 regularization: ", clf_l1.best_score_)
print("Best Gamma for logistic regression with l-1 regularization: ", best_l1_gamma)
print("Coefficients learned by logistic regression with l-1 regularization: ", clf_l1.best_estimator_.coef_)
print_classifier_metrics(y_test,y_pred_l1,name="Logistic Regression with l-1 regularization")


########################################################## QUESTION 6
from sklearn.naive_bayes import GaussianNB

gnb = GaussianNB()
y_pred_gnb = gnb.fit(X_train_LSI,y_train).predict(X_test_LSI) # training a Gaussian Naive Bayes model on training data
print_classifier_metrics(y_test,y_pred_gnb,name="Gaussian Naive Bayes")

gnb_cm = confusion_matrix(y_test,y_pred_gnb) # Gaussian Naive Bayes confusion matrix
plt.figure(); plot_confusion_matrix(gnb_cm, classes=class_names, title='Gaussian Naive Bayes Confusion Matrix');

plot_roc_curve(y_test,gnb.predict_proba(X_test_LSI)[:,1],name="Gaussian Naive Bayes") # Gaussian Naive Bayes ROC curve 


######################################################### QUESTION 7
from sklearn.pipeline import Pipeline
from tempfile import mkdtemp
from shutil import rmtree
from sklearn.externals.joblib import Memory


def with_lemmatization(doc):
    sent = lemmatize_sent(doc) # lemmatize
    sent = [i for i in sent if i not in string.punctuation] # remove punctuation
    sent = [i for i in sent if not i.isdigit()] # remove numbers
    #print("with_lemmatization")
    return sent

def without_lemmatization(doc):
    sent = nltk.word_tokenize(doc)
    sent = [i for i in sent if i not in string.punctuation] # remove punctuation
    sent = [i for i in sent if not i.isdigit()] # remove numbers
    #print("without_lemmatization")
    return sent

data_with_head_foot = fetch_20newsgroups(subset = 'train', categories = categories, shuffle = True, random_state = 42) # keeping headers and footer
data_without_head_foot = fetch_20newsgroups(subset = 'train', categories = categories, shuffle = True, random_state = 42, remove=('headers','footers')) # removing headers and footers

cachedir = mkdtemp()
memory = Memory(cachedir=cachedir, verbose=10)

pipeline = Pipeline([('vect',TfidfVectorizer(stop_words='english')),
                     ('reduce_dim',None),
                     ('classify',None)], memory=memory)
                     
param_grid = {
    'vect__min_df': (3,5),
    'vect__analyzer': (with_lemmatization, without_lemmatization),
    'reduce_dim':(TruncatedSVD(n_components=50,random_state=42), NMF(n_components=50,init='random', random_state=42)),
    'classify':(LinearSVC(C=best_svm_gamma,random_state=42), LogisticRegression(penalty='l2',C=best_l2_gamma,random_state=42),
                LogisticRegression(penalty='l1',C=best_l1_gamma,random_state=42), GaussianNB())
    } # parameter grid for the pipeline to find the best combination

grid = GridSearchCV(pipeline,cv=5,param_grid=param_grid,scoring='accuracy')

grid.fit(data_with_head_foot.data, data_with_head_foot.target)
print("Best score for pipeline with headers and footers: ", grid.best_score_)
print("Best params for pipeline with headers and footers: ", grid.best_params_)
print("Best estimator for pipeline with headers and footers: ", grid.best_estimator_)

grid.fit(data_without_head_foot.data, data_with_head_foot.target)
rmtree(cachedir)
print("Best score for pipeline without headers and footers: ", grid.best_score_)
print("Best params for pipeline without headers and footers: ", grid.best_params_)
print("Best estimator for pipeline without headers and footers: ", grid.best_estimator_)


y_pred_pipe1 = LinearSVC(C=10,random_state=42).fit(X_train_LSI,y_train).predict(X_test_LSI) # testing the best combination for dataset with headers and footers
print_classifier_metrics(y_test,y_pred_pipe1,name="Pipeline1")

train_dataset = fetch_20newsgroups(subset = 'train', categories = categories, shuffle = True, random_state = 42, remove=('headers','footers')) # training set with removed headers and footers
for n,sent in enumerate(train_dataset.data):
    sent = lemmatize_sent(sent) # lemmatize
    sent = [i for i in sent if i not in string.punctuation] # remove punctuation
    sent = [i for i in sent if not i.isdigit()] # remove numbers
    train_dataset.data[n] = " ".join(sent)

test_dataset = fetch_20newsgroups(subset = 'test', categories = categories, shuffle = True, random_state = 42, remove=('headers','footers'))  # testing set with removed headers and footers
for n,sent in enumerate(test_dataset.data):
    sent = lemmatize_sent(sent) # lemmatize
    sent = [i for i in sent if i not in string.punctuation] # remove punctuation
    sent = [i for i in sent if not i.isdigit()] # remove numbers
    test_dataset.data[n] = " ".join(sent)

tfidf_vect = TfidfVectorizer(stop_words='english',min_df=3)

X_train_tfidf = tfidf_vect.fit_transform(train_dataset.data)
X_test_tfidf = tfidf_vect.transform(test_dataset.data)

LSI = TruncatedSVD(n_components=50, random_state=42)
X_train_LSI = LSI.fit_transform(X_train_tfidf)
X_test_LSI = LSI.transform(X_test_tfidf)

y_pred_pipe2 = LogisticRegression(random_state=42,C=10).fit(X_train_LSI,y_train).predict(X_test_LSI) # testing best combination for dataset without headers and footers
print_classifier_metrics(y_test,y_pred_pipe2,name="Pipeline2")


######################################################## QUESTION 8
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier

categories = ['comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware', 'misc.forsale', 'soc.religion.christian']
class_names = ['IBM Hardware', 'Mac Hardware', 'For Sale', 'Christianity'] # shorter names for the categories

train_dataset = fetch_20newsgroups(subset = 'train', categories = categories, shuffle = True, random_state = 42) # fetching training data
for n,sent in enumerate(train_dataset.data):
    sent = lemmatize_sent(sent) # lemmatize
    sent = [i for i in sent if i not in string.punctuation] # remove punctuation
    sent = [i for i in sent if not i.isdigit()] # remove numbers
    train_dataset.data[n] = " ".join(sent)

test_dataset = fetch_20newsgroups(subset = 'test', categories = categories, shuffle = True, random_state = 42) # fetching test data
for n,sent in enumerate(test_dataset.data):
    sent = lemmatize_sent(sent) # lemmatize
    sent = [i for i in sent if i not in string.punctuation] # remove punctuation
    sent = [i for i in sent if not i.isdigit()] # remove numbers
    test_dataset.data[n] = " ".join(sent)

tfidf_vect = TfidfVectorizer(stop_words='english',min_df=3)

X_train_tfidf = tfidf_vect.fit_transform(train_dataset.data) # vectorizing data
X_test_tfidf = tfidf_vect.transform(test_dataset.data)

LSI = TruncatedSVD(n_components=50, random_state=42) # performing dimensionality reduction
X_train_LSI = LSI.fit_transform(X_train_tfidf)
X_test_LSI = LSI.transform(X_test_tfidf)

y_train = train_dataset.target
y_test = test_dataset.target

y_pred_gnb = gnb.fit(X_train_LSI,y_train).predict(X_test_LSI) # training multiclass Gaussian Naive Bayes model
print_classifier_metrics(y_test,y_pred_gnb,name="Multiclass Gaussian Naive Bayes",average='weighted')
gnb_cm = confusion_matrix(y_test,y_pred_gnb) # Multiclass Gaussian Naive Bayes confusion matrix
plt.figure(); plot_confusion_matrix(gnb_cm, classes=class_names, title='Multiclass Gaussian Naive Bayes Confusion Matrix') # Multiclass Gaussian Naive Bayes roc curve

params = {'estimator__C':[0.001,0.01,0.1,1,10,100,1000]} # parameters to try for best gamma in Linear SVC

svm_one = OneVsOneClassifier(LinearSVC(random_state=42)) # One v One SVM classifier 
clf_one = GridSearchCV(svm_one,params,cv=5,scoring='accuracy') # grid search to find best gamma
y_pred_one = clf_one.fit(X_train_LSI, y_train).best_estimator_.predict(X_test_LSI)
print(clf_one.best_estimator_)
print_classifier_metrics(y_test,y_pred_one,name="1v1 SVM",average='weighted')
one_cm = confusion_matrix(y_test,y_pred_one) # One v One SVM confusion matrix
plt.figure(); plot_confusion_matrix(one_cm, classes=class_names, title='1v1 SVM Confusion Matrix')

svm_rest = OneVsRestClassifier(LinearSVC(random_state=42)) # One v Rest SVM classifier
clf_rest = GridSearchCV(svm_rest,params,cv=5,scoring='accuracy') # grid search to find best gamma
y_pred_rest = clf_rest.fit(X_train_LSI, y_train).best_estimator_.predict(X_test_LSI)
print(clf_rest.best_estimator_)
print_classifier_metrics(y_test,y_pred_rest,name="1vRest SVM",average='weighted')
rest_cm = confusion_matrix(y_test,y_pred_rest) # One v Rest SVM confusion matrix
plt.figure(); plot_confusion_matrix(rest_cm, classes=class_names, title='1vRest SVM Confusion Matrix')


plt.show()
