import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.cluster as cluster
import sklearn.metrics as metrics
import sklearn.preprocessing
import scipy.cluster.hierarchy as sch
import sklearn.mixture as mixture
import pyclustertend 
import random
from sklearn.cluster import KMeans, MeanShift, estimate_bandwidth
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.metrics import silhouette_score
from sklearn.model_selection import train_test_split
from kmodes.kprototypes import KPrototypes
from yellowbrick.cluster import KElbowVisualizer, SilhouetteVisualizer, InterclusterDistance
from kneed import KneeLocator
from sklearn import datasets
from sklearn import tree
from sklearn import metrics
from sklearn.model_selection import cross_val_predict
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
from sklearn.datasets import make_blobs 
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
import matplotlib.cm as cm
import sys
import warnings
from sklearn.svm import SVC
import timeit
import scipy.stats as stats
from scipy.special import expit
import pylab
from scipy.stats import shapiro
from sklearn.datasets import make_classification
from sklearn.tree import export_graphviz
from statsmodels.stats.outliers_influence import variance_inflation_factor


from mpl_toolkits.mplot3d import Axes3D
plt.rcParams['figure.figsize'] = (16, 9)
plt.style.use('ggplot')

Data = pd.read_csv('train.csv')

#Test = pd.read_csv('test.csv')

#print(Data)

normal = Data.select_dtypes(include = np.number)
normal = normal.dropna()
r = ''
"""
fig = plt.figure()
g = 0
for i in CN:
    estadistico1, p_value1 = stats.kstest(normal[i], 'norm')

    if p_value1 > 0.5:
        r = 'Es normal'
    else:
        r = 'no es normal'

    plt.subplot(7,7,g+1)
    sns.distplot(normal[i])
    plt.xlabel(i)
    g+= 1

    print(i, ": ", r)

"""

#print(normal.describe())

J = normal.drop(['SalePrice'], axis = 1)
VIF_Data = pd.DataFrame()

VIF_Data['feature'] = J.columns

VIF_Data['VIF'] = [variance_inflation_factor(J.values, i) for i in range(len(J.columns))]

print(VIF_Data)


normal = normal.drop(['Id', 'LowQualFinSF', 'YearBuilt', 'YearRemodAdd', 'GarageYrBlt', 'FullBath', 'MoSold', 'YrSold', 'MSSubClass', 'OverallCond', 'BsmtFinSF1','BsmtFinSF2', 'BsmtUnfSF', '2ndFlrSF', 'BsmtFullBath', 'BsmtHalfBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'WoodDeckSF', 'EnclosedPorch','3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal', 'LotFrontage', 'LotArea', 'MasVnrArea', '2ndFlrSF', 'TotRmsAbvGrd', 'Fireplaces', 'OpenPorchSF' ], axis = 1)

CN = normal.columns.values
print(CN)
correlation_mat = normal.corr()
NC = normal.columns.values

SP = correlation_mat.iloc[-1]

SaleP = normal[['SalePrice']]



#print(SP)

#variables = correlation_mat.query("sector == 'SalePrice & ")
#print(correlation_mat.iloc[-1])


#sns.heatmap(correlation_mat, annot = True)
#plt.tight_layout()
#plt.show()

CN = normal.columns.values

#print(CN)





# In[268]:

"""
n = 0
while n < 20:
    for i in CN:
        normal = normal[(normal[i] < normal[i].mean()+2*(normal[i].std())) & (normal[i] > normal[i].mean()-2*(normal[i].std()))] 
        n += 1


fig = plt.figure()
g = 0
for i in CN:
	plt.subplot(5,3,g+1)
	sns.distplot(normal[i])
	plt.xlabel(i)
	g += 1

"""
#normal = normal[['OverallQual', 'TotalBsmtSF', 'GrLivArea' ,'FullBath', 'GarageCars', 'SalePrice']]
H = normal

X = np.array(normal)
X.shape
print('Hopkins', pyclustertend.hopkins(X,len(X)))


"""

print(X.shape)
X_scale=sklearn.preprocessing.scale(X)
