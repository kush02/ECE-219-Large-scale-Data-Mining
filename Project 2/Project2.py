"""
    Kushagra Rastogi
    304640248
    ECE 219
    Project 2
"""

import numpy as np
np.random.seed(42)
import random
random.seed(42)

import nltk, string
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_20newsgroups


categories = ['comp.sys.ibm.pc.hardware', 'comp.graphics','comp.sys.mac.hardware', 'comp.os.ms-windows.misc','rec.autos', 'rec.motorcycles','rec.sport.baseball', 'rec.sport.hockey']
dataset = fetch_20newsgroups(subset='all', categories=categories,shuffle=True, random_state=42)


####################################### QUESTION 1
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf_vect = TfidfVectorizer(stop_words='english',min_df=3)
X_train_tfidf = tfidf_vect.fit_transform(dataset.data) # making the tfidf matrix
print("Shape of TF-IDF matrix: ", X_train_tfidf.shape)


####################################### QUESTION 2
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import contingency_matrix, homogeneity_score, v_measure_score, completeness_score, adjusted_rand_score, adjusted_mutual_info_score

y_true = [int(i/4) for i in dataset.target]

km = KMeans(n_clusters=2, random_state=0, max_iter=1000, n_init=30)
y_pred = km.fit_predict(X_train_tfidf)
con_mat = contingency_matrix(y_true,y_pred)
print("Contingency table: ", con_mat)


####################################### QUESTION 3

print("Homogeneity score: ", homogeneity_score(y_true,y_pred))
print("Completeness score: ",completeness_score(y_true,y_pred))
print("V-measure score: ",v_measure_score(y_true,y_pred))
print("Adjusted Rand score: ",adjusted_rand_score(y_true,y_pred))
print("Adjusted mutual information score: ",adjusted_mutual_info_score(y_true,y_pred))


######################################## QUESTION 4
from sklearn.decomposition import TruncatedSVD

svd = TruncatedSVD(n_components=1000,random_state=42)
X_train_svd = svd.fit_transform(X_train_tfidf)
plt.figure()
plt.plot(np.arange(1000)+1,sorted(svd.explained_variance_ratio_,reverse=True))
plt.scatter(np.arange(1000)+1,sorted(svd.explained_variance_ratio_,reverse=True),)
plt.xlabel("Components"); plt.ylabel("Explained Variance Ratio per Component")

plt.figure()
plt.plot(np.arange(1000)+1,np.cumsum(svd.explained_variance_ratio_))
plt.scatter(np.arange(1000)+1,np.cumsum(svd.explained_variance_ratio_))
plt.xlabel("Components"); plt.ylabel("Total Explained Variance Ratio")


######################################### QUESTION 5
from sklearn.decomposition import NMF

r = [1,2,3,5,10,20,50,100,300]
hom_score = []; complt_score = []; v_score = []; adj_rand_score = []; adj_mut_inf_score = []
for i in r:
    y_pred = km.fit_predict(TruncatedSVD(n_components=i,random_state=42).fit_transform(X_train_tfidf))
    hom_score.append(homogeneity_score(y_true,y_pred))
    complt_score.append(completeness_score(y_true,y_pred))
    v_score.append(v_measure_score(y_true,y_pred))
    adj_rand_score.append(adjusted_rand_score(y_true,y_pred))
    adj_mut_inf_score.append(adjusted_mutual_info_score(y_true,y_pred))

fig, ax = plt.subplots()
ax.plot(r,hom_score, 'r', label='Homogeneity score')
ax.plot(r, complt_score, 'b', label='Completeness score')
ax.plot(r, v_score, 'g', label='V-measure score')
ax.plot(r,adj_rand_score,'y',label='Adjusted Rand score')
ax.plot(r,adj_mut_inf_score,'m',label='Adjusted Mutual Information score')
ax.legend(loc='best')
plt.xlabel("Number of components"); plt.ylabel("Score"); plt.title("Measure scores for SVD")
print("SVD")
print(hom_score)
print(complt_score)
print(v_score)
print(adj_rand_score)
print(adj_mut_inf_score)

hom_score = []; complt_score = []; v_score = []; adj_rand_score = []; adj_mut_inf_score = []
for i in r:
    y_pred = km.fit_predict(NMF(n_components=i,init='random',random_state=42).fit_transform(X_train_tfidf))
    hom_score.append(homogeneity_score(y_true,y_pred))
    complt_score.append(completeness_score(y_true,y_pred))
    v_score.append(v_measure_score(y_true,y_pred))
    adj_rand_score.append(adjusted_rand_score(y_true,y_pred))
    adj_mut_inf_score.append(adjusted_mutual_info_score(y_true,y_pred))

fig, ax = plt.subplots()
ax.plot(r,hom_score, 'r', label='Homogeneity score')
ax.plot(r, complt_score, 'b', label='Completeness score')
ax.plot(r, v_score, 'g', label='V-measure score')
ax.plot(r,adj_rand_score,'y',label='Adjusted Rand Index')
ax.plot(r,adj_mut_inf_score,'m',label='Adjusted Mutual Information score')
ax.legend(loc='best')
plt.xlabel("Number of components"); plt.ylabel("Score"); plt.title("Measure score for NMF")
print("NMF")
print(hom_score)
print(complt_score)
print(v_score)
print(adj_rand_score)
print(adj_mut_inf_score)


######################################## QUESTION 6



######################################## QUESTION 7
from sklearn.decomposition import NMF
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans

r_best_svd = 2
r_best_nmf = 2

reduced_data_svd = TruncatedSVD(n_components=r_best_svd,random_state=0).fit_transform(X_train_tfidf)
plt.figure()
plt.scatter(reduced_data_svd[:,0],reduced_data_svd[:,1],c=y_true,cmap='viridis')
plt.title("SVD Ground truth class labels (r=2)"); plt.legend()

km = KMeans(n_clusters=2, random_state=0, max_iter=1000, n_init=30)
svd_labels = km.fit_predict(reduced_data_svd)
plt.figure()
plt.scatter(reduced_data_svd[:,0],reduced_data_svd[:,1],c=svd_labels,cmap='viridis')
plt.title("SVD Clustering class labels (r=2)"); plt.legend()

reduced_data_nmf = NMF(n_components=r_best_nmf,init='random',random_state=0).fit_transform(X_train_tfidf)
plt.figure()
plt.scatter(reduced_data_nmf[:,0],reduced_data_nmf[:,1],c=y_true,cmap='viridis')
plt.title("NMF Ground truth class labels (r=2)"); plt.legend()

nmf_labels = km.fit_predict(reduced_data_nmf)
plt.figure()
plt.scatter(reduced_data_nmf[:,0],reduced_data_nmf[:,1],c=nmf_labels,cmap='viridis')
plt.title("NMF Clustering class labels (r=2)"); plt.legend()


########################################## QUESTION 8
from sklearn import preprocessing

def log_transform(X):
    m,n = X.shape
    A = np.zeros((m,n))
    c = 0.01
    for i,row in enumerate(X):
        for j,col in enumerate(row):
            A[i][j] = np.sign(col)*(np.log10(np.abs(col)+c)-np.log10(c))
    return A

svd_log = log_transform(reduced_data_svd)
svd_log = preprocessing.scale(svd_log)
svd_log_labels = km.fit_predict(svd_log)
plt.figure()
plt.scatter(svd_log[:,0],svd_log[:,1],c=svd_log_labels,cmap='viridis')
plt.title("SVD (Log->Scaled) Clustering class labels"); plt.legend()

nmf_log = preprocessing.scale(reduced_data_nmf)
nmf_log = log_transform(nmf_log)
nmf_log_labels = km.fit_predict(nmf_log)
plt.figure()
plt.scatter(nmf_log[:,0],nmf_log[:,1],c=nmf_log_labels,cmap='viridis')
plt.title("NMF (Scaled->Log) Clustering class labels"); plt.legend()


######################################### QUESTION 9



######################################### QUESTION 10
from sklearn.metrics.cluster import homogeneity_score, v_measure_score, completeness_score, adjusted_rand_score, adjusted_mutual_info_score

y_pred = km.fit_predict(svd_log)
print("Homogeneity score for SVD (Log->Scaled): ", homogeneity_score(y_true,y_pred))
print("Completeness score for SVD (Log->Scaled): ",completeness_score(y_true,y_pred))
print("V-measure score for SVD (Log->Scaled): ",v_measure_score(y_true,y_pred))
print("Adjusted Rand score for SVD (Log->Scaled): ",adjusted_rand_score(y_true,y_pred))
print("Adjusted mutual information score for SVD (Log->Scaled): ",adjusted_mutual_info_score(y_true,y_pred))

y_pred = km.fit_predict(nmf_log)
print("Homogeneity score for NMF (Log->Scaled): ", homogeneity_score(y_true,y_pred))
print("Completeness score for NMF (Log->Scaled): ",completeness_score(y_true,y_pred))
print("V-measure score for NMF (Log->Scaled): ",v_measure_score(y_true,y_pred))
print("Adjusted Rand score for NMF (Log->Scaled): ",adjusted_rand_score(y_true,y_pred))
print("Adjusted mutual information score for NMF (Log->Scaled): ",adjusted_mutual_info_score(y_true,y_pred))


######################################### QUESTION 11

dataset = fetch_20newsgroups(subset='all',shuffle=True, random_state=42)
tfidf_vect = TfidfVectorizer(stop_words='english',min_df=3)
X_train_tfidf = tfidf_vect.fit_transform(dataset.data) # making the tfidf matrix
print(X_train_tfidf.shape)

y_true = dataset.target

km = KMeans(n_clusters=20, random_state=0, max_iter=1000, n_init=30)
y_pred = km.fit_predict(X_train_tfidf)
con_mat = contingency_matrix(y_true,y_pred)
print("Contingency table: ")
print(con_mat)

print("Homogeneity score for the whole corpus: ", homogeneity_score(y_true,y_pred))
print("Completeness score for the whole corpus: ",completeness_score(y_true,y_pred))
print("V-measure score for the whole corpus: ",v_measure_score(y_true,y_pred))
print("Adjusted Rand score for the whole corpus: ",adjusted_rand_score(y_true,y_pred))
print("Adjusted mutual information score for the whole corpus: ",adjusted_mutual_info_score(y_true,y_pred))



######################################## QUESTION 12
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import contingency_matrix, homogeneity_score, v_measure_score, completeness_score, adjusted_rand_score, adjusted_mutual_info_score
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import NMF
from sklearn import preprocessing


dataset = fetch_20newsgroups(subset='all',shuffle=True, random_state=42)
tfidf_vect = TfidfVectorizer(stop_words='english',min_df=3)
X_train_tfidf = tfidf_vect.fit_transform(dataset.data) # making the tfidf matrix

y_true = dataset.target

km = KMeans(n_clusters=20, random_state=0, max_iter=1000, n_init=30)

r = [1,2,3,5,10,20,50,100,300]
hom_score = []; complt_score = []; v_score = []; adj_rand_score = []; adj_mut_inf_score = []
for i in r:
    y_pred = km.fit_predict(TruncatedSVD(n_components=i,random_state=42).fit_transform(X_train_tfidf))
    hom_score.append(homogeneity_score(y_true,y_pred))
    complt_score.append(completeness_score(y_true,y_pred))
    v_score.append(v_measure_score(y_true,y_pred))
    adj_rand_score.append(adjusted_rand_score(y_true,y_pred))
    adj_mut_inf_score.append(adjusted_mutual_info_score(y_true,y_pred))

fig, ax = plt.subplots()
ax.plot(r,hom_score, 'r', label='Homogeneity score')
ax.plot(r, complt_score, 'b', label='Completeness score')
ax.plot(r, v_score, 'g', label='V-measure score')
ax.plot(r,adj_rand_score,'y',label='Adjusted Rand score')
ax.plot(r,adj_mut_inf_score,'m',label='Adjusted Mutual Information score')
ax.legend(loc='best')
plt.xlabel("Number of components"); plt.ylabel("Score"); plt.title("Measure scores for SVD")
print("SVD")
print(hom_score)
print(complt_score)
print(v_score)
print(adj_rand_score)
print(adj_mut_inf_score)

hom_score = []; complt_score = []; v_score = []; adj_rand_score = []; adj_mut_inf_score = []
for i in r:
    y_pred = km.fit_predict(NMF(n_components=i,init='random',random_state=42).fit_transform(X_train_tfidf))
    hom_score.append(homogeneity_score(y_true,y_pred))
    complt_score.append(completeness_score(y_true,y_pred))
    v_score.append(v_measure_score(y_true,y_pred))
    adj_rand_score.append(adjusted_rand_score(y_true,y_pred))
    adj_mut_inf_score.append(adjusted_mutual_info_score(y_true,y_pred))

fig, ax = plt.subplots()
ax.plot(r,hom_score, 'r', label='Homogeneity score')
ax.plot(r, complt_score, 'b', label='Completeness score')
ax.plot(r, v_score, 'g', label='V-measure score')
ax.plot(r,adj_rand_score,'y',label='Adjusted Rand Index')
ax.plot(r,adj_mut_inf_score,'m',label='Adjusted Mutual Information score')
ax.legend(loc='best')
plt.xlabel("Number of components"); plt.ylabel("Score"); plt.title("Measure score for NMF")
print("NMF")
print(hom_score)
print(complt_score)
print(v_score)
print(adj_rand_score)
print(adj_mut_inf_score)


r_best_svd = 300
r_best_nmf = 300

reduced_data_svd = TruncatedSVD(n_components=r_best_svd,random_state=0).fit_transform(X_train_tfidf)

svd_log = log_transform(reduced_data_svd)
y_pred = km.fit_predict(svd_log)
print("Metrics for SVD (Log)")
print("Homogeneity score for SVD (Log): ", homogeneity_score(y_true,y_pred))
print("Completeness score for SVD (Log): ",completeness_score(y_true,y_pred))
print("V-measure score for SVD (Log): ",v_measure_score(y_true,y_pred))
print("Adjusted Rand score for SVD (Log): ",adjusted_rand_score(y_true,y_pred))
print("Adjusted mutual information score for SVD (Log): ",adjusted_mutual_info_score(y_true,y_pred))

svd_scale = preprocessing.scale(reduced_data_svd)
y_pred = km.fit_predict(svd_scale)
print("Metrics for SVD (Scaled)")
print("Homogeneity score for SVD (Scaled): ", homogeneity_score(y_true,y_pred))
print("Completeness score for SVD (Scaled): ",completeness_score(y_true,y_pred))
print("V-measure score for SVD (Scaled): ",v_measure_score(y_true,y_pred))
print("Adjusted Rand score for SVD (Scaled): ",adjusted_rand_score(y_true,y_pred))
print("Adjusted mutual information score for SVD (Scaled): ",adjusted_mutual_info_score(y_true,y_pred))

svd_log = log_transform(reduced_data_svd)
svd_log = preprocessing.scale(svd_log)
y_pred = km.fit_predict(svd_log)
print("Metrics for SVD (Log->Scaled)")
print("Homogeneity score for SVD (Log->Scaled): ", homogeneity_score(y_true,y_pred))
print("Completeness score for SVD (Log->Scaled): ",completeness_score(y_true,y_pred))
print("V-measure score for SVD (Log->Scaled): ",v_measure_score(y_true,y_pred))
print("Adjusted Rand score for SVD (Log->Scaled): ",adjusted_rand_score(y_true,y_pred))
print("Adjusted mutual information score for SVD (Log->Scaled): ",adjusted_mutual_info_score(y_true,y_pred))

svd_scale = preprocessing.scale(reduced_data_svd)
svd_scale = log_transform(svd_scale)
y_pred = km.fit_predict(svd_scale)
print("Metrics for SVD (Scaled->Log)")
print("Homogeneity score for SVD (Scaled->Log): ", homogeneity_score(y_true,y_pred))
print("Completeness score for SVD (Scaled->Log): ",completeness_score(y_true,y_pred))
print("V-measure score for SVD (Scaled->Log): ",v_measure_score(y_true,y_pred))
print("Adjusted Rand score for SVD (Scaled->Log): ",adjusted_rand_score(y_true,y_pred))
print("Adjusted mutual information score for SVD (Scaled->Log): ",adjusted_mutual_info_score(y_true,y_pred))


reduced_data_nmf = NMF(n_components=r_best_nmf,init='random',random_state=0).fit_transform(X_train_tfidf)

nmf_log = log_transform(reduced_data_nmf)
y_pred = km.fit_predict(nmf_log)
print("Metrics for NMF (Log)")
print("Homogeneity score for NMF (Log): ", homogeneity_score(y_true,y_pred))
print("Completeness score for NMF (Log): ",completeness_score(y_true,y_pred))
print("V-measure score for NMF (Log): ",v_measure_score(y_true,y_pred))
print("Adjusted Rand score for NMF (Log): ",adjusted_rand_score(y_true,y_pred))
print("Adjusted mutual information score for NMF (Log): ",adjusted_mutual_info_score(y_true,y_pred))

nmf_scale = preprocessing.scale(reduced_data_nmf)
y_pred = km.fit_predict(nmf_scale)
print("Metrics for NMF (Scaled)")
print("Homogeneity score for NMF (Scaled): ", homogeneity_score(y_true,y_pred))
print("Completeness score for NMF (Scaled): ",completeness_score(y_true,y_pred))
print("V-measure score for NMF (Scaled): ",v_measure_score(y_true,y_pred))
print("Adjusted Rand score for NMF (Scaled): ",adjusted_rand_score(y_true,y_pred))
print("Adjusted mutual information score for NMF (Scaled): ",adjusted_mutual_info_score(y_true,y_pred))

nmf_log = log_transform(reduced_data_nmf)
nmf_log = preprocessing.scale(nmf_log)
y_pred = km.fit_predict(nmf_log)
print("Metrics for NMF (Log->Scaled)")
print("Homogeneity score for NMF (Log->Scaled): ", homogeneity_score(y_true,y_pred))
print("Completeness score for NMF (Log->Scaled): ",completeness_score(y_true,y_pred))
print("V-measure score for NMF (Log->Scaled): ",v_measure_score(y_true,y_pred))
print("Adjusted Rand score for NMF (Log->Scaled): ",adjusted_rand_score(y_true,y_pred))
print("Adjusted mutual information score for NMF (Log->Scaled): ",adjusted_mutual_info_score(y_true,y_pred))

nmf_scale = preprocessing.scale(reduced_data_nmf)
nmf_scale = log_transform(nmf_scale)
y_pred = km.fit_predict(nmf_scale)
print("Metrics for NMF (Scaled->Log)")
print("Homogeneity score for NMF (Scaled->Log): ", homogeneity_score(y_true,y_pred))
print("Completeness score for NMF (Scaled->Log): ",completeness_score(y_true,y_pred))
print("V-measure score for NMF (Scaled->Log): ",v_measure_score(y_true,y_pred))
print("Adjusted Rand score for NMF (Scaled->Log): ",adjusted_rand_score(y_true,y_pred))
print("Adjusted mutual information score for NMF (Scaled->Log): ",adjusted_mutual_info_score(y_true,y_pred))


plt.show()
