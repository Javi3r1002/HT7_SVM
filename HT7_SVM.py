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



pyclustertend.vat(X_scale)
pyclustertend.vat(X)
plt.show()
"""
"""
numeroClusters = range(1,11)
wcss = []
for i in numeroClusters:
    kmeans = cluster.KMeans(n_clusters=i)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

plt.plot(numeroClusters, wcss)
plt.xlabel("Número de clusters")
plt.ylabel("Puntuación")
plt.title("Gráfico de Codo")


"""

"""
rango_n_clusters = [2, 3, 4, 5]

for n_clusters in rango_n_clusters:
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches(18, 7)
    ax1.set_xlim([-0.1, 1])
    ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])
    clusterer = KMeans(n_clusters=n_clusters, random_state=10)
    cluster_labels = clusterer.fit_predict(X)
    silhouette_avg = silhouette_score(X, cluster_labels)
    print(
        "For n_clusters =",
        n_clusters,
        "The average silhouette_score is :",
        silhouette_avg,
    )

    sample_silhouette_values = silhouette_samples(X, cluster_labels)

    y_lower = 10
    for i in range(n_clusters):
        ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]
        ith_cluster_silhouette_values.sort()
        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i
        color = cm.nipy_spectral(float(i) / n_clusters)
        ax1.fill_betweenx(
            np.arange(y_lower, y_upper),
            0,
            ith_cluster_silhouette_values,
            facecolor=color,
            edgecolor=color,
            alpha=0.7,
        )
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
        y_lower = y_upper + 10  # 10 for the 0 samples

    ax1.set_title("The silhouette plot for the various clusters.")
    ax1.set_xlabel("The silhouette coefficient values")
    ax1.set_ylabel("Cluster label")
    ax1.axvline(x=silhouette_avg, color="red", linestyle="--")
    ax1.set_yticks([])  # Clear the yaxis labels / ticks
    ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])
    colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
    ax2.scatter(
        X[:, -1], X[:, 0], marker=".", s=30, lw=0, alpha=0.7, c=colors, edgecolor="k"
    )
    centers = clusterer.cluster_centers_
    ax2.scatter(
        centers[:, -1],
        centers[:, 0],
        marker="o",
        c="white",
        alpha=1,
        s=200,
        edgecolor="k",
    )

    for i, c in enumerate(centers):
        ax2.scatter(c[-1], c[0], marker="$%d$" % i, alpha=1, s=50, edgecolor="k")

    ax2.set_title("The visualization of the clustered data.")
    ax2.set_xlabel("Feature space for the 1st feature")
    ax2.set_ylabel("Feature space for the 2nd feature")

    plt.suptitle(
        "Silhouette analysis for KMeans clustering on sample data with n_clusters = %d"
        % n_clusters,
        fontsize=14,
        fontweight="bold",
    )
"""

km = cluster.KMeans(n_clusters=3, random_state = 5).fit(X)
centroides = km.cluster_centers_
#print(centroides)


normal = km.predict(X)
#plt.scatter(X[normal == 0, -1], X[normal == 0, -1],s=100,c='red', label = "Cluster 1")
Cluster_bajo = X[normal == 0, -1]
Cluster_bajo = Cluster_bajo.tolist()
print('máximo primer cluster (rojo)', max(Cluster_bajo))
print('mínimo primer cluster (rojo)',min(Cluster_bajo))
Barato = H[((H['SalePrice']>(min(Cluster_bajo)))& (H['SalePrice']<max(Cluster_bajo)))]

#plt.scatter(X[normal == 1, -1], X[normal == 1, -1],s=100,c='blue', label = "Cluster 2")
Cluster_medio = X[normal == 1, -1]
Cluster_medio = Cluster_medio.tolist()
print('máximo segundo cluster (azul)', max(Cluster_medio))
print('mínimo segundo cluster (azul)',min(Cluster_medio))
Medio = H[((H['SalePrice']>(min(Cluster_medio))) & (H['SalePrice']<max(Cluster_medio)))]

#plt.scatter(X[normal == 2, -1], X[normal == 2, -1],s=100,c='green', label = "Cluster 3")
Cluster_alto = X[normal == 2, -1]
Cluster_alto = Cluster_alto.tolist()
print('máximo tercer cluster (verde)', max(Cluster_alto))
print('mínimo tercer cluster (verde)',min(Cluster_alto))
Alto =  H[((H['SalePrice']>(min(Cluster_alto))) & (H['SalePrice']<max(Cluster_alto)))]

#plt.scatter(km.cluster_centers_[:,-1],km.cluster_centers_[:,-1], s=300, c="yellow",marker="*", label="Centroides")
#plt.title("Grupo casa")
#plt.xlabel("Precio de venta")
#plt.ylabel("Precio de venta")
#plt.legend()


#plt.show()



CH = H.columns.values

Cate = []
for row in H['SalePrice']:
    if row in Cluster_bajo : Cate.append(1)
    elif row in Cluster_medio:   Cate.append(2)
    elif row in Cluster_alto:  Cate.append(3)

    
    else:
        print('dndofinoid')
H['Categoria'] = Cate
print(H.groupby('Categoria').size())
G = H
H = H.drop(['SalePrice'], axis = 1)
#sns.pairplot(H, hue="Categoria",  diag_kws={'bw': 100})
#plt.tight_layout()
#plt.show()
"""
print(Barato.describe().transpose())
print(Medio.describe().transpose())
print(Alto.describe().transpose())
"""
#Modelo para casas baratas

H = G
Cate = []
for row in H['SalePrice']:
    if row in Cluster_bajo : Cate.append(0)
    elif row in Cluster_medio:   Cate.append(1)
    elif row in Cluster_alto:  Cate.append(2)

    
    else:
    	print('dndofinoid')
H['Categoria'] = Cate
H = H.drop(['SalePrice'], axis = 1)

print(H.groupby('Categoria').size())
H['Categoria'] = H['Categoria'].astype('category')


y = H.pop("Categoria") #La variable respuesta
X = H #El resto de los datos

random.seed(123)

X_train, X_test,y_train, y_test = train_test_split(X, y,test_size=0.3,train_size=0.7, random_state = 123)

#Modelo 1: Se utilizan todos los parámetros en default

clf_svm = SVC(random_state=123)
start = timeit.default_timer()
clf_svm.fit(X_train,y_train)
end = timeit.default_timer()
print('Tiempo de Fit para el modelo con los parámetros default: ',  end-start)

y_predf = clf_svm.predict(X_train)
start = timeit.default_timer()
y_pred = clf_svm.predict(X_test)
end = timeit.default_timer()
print('Tiempo de predict del modelo con los parámetros default: ',  end-start)
cm = confusion_matrix(y_test,y_pred)
print ("Accuracy entrenamiento para el modelo con los parámetros default:",metrics.accuracy_score(y_train, y_predf))
print ("Accuracy para el modelo con los parámetros default:",metrics.accuracy_score(y_test, y_pred))
print ("Precision:", metrics.precision_score(y_test,y_pred,average='weighted') )
print ("Recall: ", metrics.recall_score(y_test,y_pred,average='weighted'))
print("Matriz de confusión del modelo con los parámetros default", '\n',cm)




#Modelo 2: Se utiliza un kernel poly con un grado de 3
clf_svm = SVC(kernel = 'poly', degree = 3, random_state=123)
start = timeit.default_timer()
clf_svm.fit(X_train,y_train)
end = timeit.default_timer()
print('Tiempo de Fit para el modelo con un kernel poly y un grado 3: ',  end-start)

y_predf = clf_svm.predict(X_train)
start = timeit.default_timer()
y_pred = clf_svm.predict(X_test)
end = timeit.default_timer()
print('Tiempo de predict del modelo con un kernel poly y un grado 3: ',  end-start)
cm = confusion_matrix(y_test,y_pred)
print ("Accuracy entrenamiento para el modelo con un kernel poly y un grado 3:",metrics.accuracy_score(y_train, y_predf))
print ("Accuracy para el modelo con un kernel poly y un grado 3:",metrics.accuracy_score(y_test, y_pred))
print ("Precision:", metrics.precision_score(y_test,y_pred,average='weighted') )
print ("Recall: ", metrics.recall_score(y_test,y_pred,average='weighted'))
print("Matriz de confusión del modelo con un kernel poly y un grado 3", '\n',cm)

#Modelo 3: Se varía el valor de C
clf_svm = SVC(C = .01, random_state=123)
start = timeit.default_timer()
clf_svm.fit(X_train,y_train)
end = timeit.default_timer()
print('Tiempo de Fit para el modelo con un kernel poly y un grado 3: ',  end-start)

y_predf = clf_svm.predict(X_train)
start = timeit.default_timer()
y_pred = clf_svm.predict(X_test)
end = timeit.default_timer()
print('Tiempo de predict del modelo con un valor de C de .01: ',  end-start)
cm = confusion_matrix(y_test,y_pred)
print ("Accuracy entrenamiento para el modelo con un valor de C de .01:",metrics.accuracy_score(y_train, y_predf))
print ("Accuracy para el modelo con un valor de C de .01:",metrics.accuracy_score(y_test, y_pred))
print ("Precision:", metrics.precision_score(y_test,y_pred,average='weighted') )
print ("Recall: ", metrics.recall_score(y_test,y_pred,average='weighted'))
print("Matriz de confusión del modelo con un valor de C de .01", '\n',cm)

# Se dejaron de eliminar las variables ,'TotalBsmtSF','1stFlrSF',
