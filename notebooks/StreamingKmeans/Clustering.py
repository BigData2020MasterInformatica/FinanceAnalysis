'''
Jose Gómez Baco
Leticia Yepez Chavez
Nadia Carrera Chahir
'''


# Required libraries
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler
from pyspark.context import SparkContext
from pyspark.sql.session import SparkSession
from pyspark.ml.clustering import KMeans, BisectingKMeans
from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark import SparkConf, SparkContext


from sklearn.manifold import TSNE #T-Distributed Stochastic Neighbor Embedding
import plotly.graph_objs as go
import pandas as pd
from pyspark.sql.types import StructType
from plotly.offline import plot, iplot
import plotly.figure_factory as ff


#Functions
def K_Means(n_clusters, dataset):
    kmeans = KMeans().setK(n_clusters).setSeed(1)
    model = kmeans.fit(dataset)

    predictions = model.transform(dataset)
    print(predictions.columns)

    evaluator = ClusteringEvaluator()

    silhouette = evaluator.evaluate(predictions)
    print("For n_clusters = " + str(n_clusters) + " Silhouette with squared euclidean distance = " + str(silhouette))

    ctr = []
    centers = model.clusterCenters()
    return model, predictions, silhouette


def BisectingK_Means(nclusters, dataset):
    bkm = BisectingKMeans().setK(n_clusters).setSeed(1)
    model = bkm.fit(dataset)

    predictions = model.transform(dataset)

    evaluator = ClusteringEvaluator()

    silhouette = evaluator.evaluate(predictions)
    print("For n_clusters = " + str(n_clusters) + " Silhouette with squared euclidean distance = " + str(silhouette))

    centers = model.clusterCenters()

    return model, predictions, silhouette


#########################################################################################################################
spark_conf = SparkConf()
spark_context = SparkContext(conf=spark_conf)
spark = SparkSession(spark_context)



path = "<ADD_FILEPATH>"
KMmodelPath = "models/modelKM/"
BKMmodelPath = "models/modelBKM/"

namefile = "data.csv"
namefilePredict = "data2020.csv"



# dataset = spark.read.format("libsvm").load("C:/Users/nadia/Documents/Universidad/2year/BigData/datachangeOverTime.txt")
#dataset = spark.read.format("libsvm").load(path + namefile)
#dataset = spark.read.format("csv").load(path + namefile, inferSchema="true", header="true", sep = ",")

fileSchema = StructType().add('symbol', 'string').add('date', 'timestamp').add('open', 'double').add('high', 'double').add('low', 'double').add('close', 'double').add('adjClose', 'double') \
    .add('volume', 'double').add('unadjustedVolume', 'double').add('change', 'double').add('changePercent', 'double').add('vwap', 'double').add('label', 'timestamp').add('year', 'double').add('changeOverTime', 'double')

dataset = spark.read.option("header", "true").option('sep', ',').option('includeTimestamp', 'true') \
        .option("timestampFormat", "yyyy-MM-dd") \
        .schema(fileSchema) \
        .csv(path + namefile)


'''total_silhouette = []
for n_clusters in range(3, 10):
    print("KMeans cluster centers for " + str(n_clusters) + " nclusters")
    ctr = []
    centers, modelKM, predictions, silhouette = K_Means(n_clusters, dataset)

    try:
        modelKM.save(path + KMmodelPath + "modelKM" + str(n_clusters))
    except:
        modelKM.write().overwrite().save(path + KMmodelPath + "modelKM" + str(n_clusters))
    
    total_silhouette.append(silhouette)'''



print("\n")

'''for n_clusters in range(3, 10):
    print("Bisecting k-means cluster centers for " + str(n_clusters) + " nclusters")
    centers, modelBKM = BisectingK_Means(n_clusters, dataset)

    try:
        modelBKM.save(path + BKMmodelPath + "modelBKM" + str(n_clusters))
    except:
        modelBKM.write().overwrite().save(path + BKMmodelPath + "modelBKM" + str(n_clusters))
    
    for center in centers:
        print(center)
    print("\n")'''


cols = ["high", "low", "open", "adjClose", "volume", "unadjustedVolume", "change", "changePercent", "changeOverTime"]
assembler = VectorAssembler(inputCols=cols, outputCol="features")
featureDf = assembler.transform(dataset)

##
total_silhouette_KM = []
for n_clusters in range(3, 10):
    print("KMeans cluster for " + str(n_clusters) + " nclusters")
    modelKM, predictionsKM, silhouetteKM = K_Means(n_clusters, featureDf)

    try:
        modelKM.save(path + KMmodelPath + "modelKM" + str(n_clusters))
    except:
        modelKM.write().overwrite().save(path + KMmodelPath + "modelKM" + str(n_clusters))
    
    total_silhouette_KM.append(silhouetteKM)
##
print("\n")
##
total_silhouette_BKM = []
for n_clusters in range(3, 10):
    print("Bisecting k-means cluster for " + str(n_clusters) + " nclusters")
    modelBKM, predictionsBKM, silhouetteBKM = BisectingK_Means(n_clusters, featureDf)

    try:
        modelBKM.save(path + BKMmodelPath + "modelBKM" + str(n_clusters))
    except:
        modelBKM.write().overwrite().save(path + BKMmodelPath + "modelBKM" + str(n_clusters))

    total_silhouette_BKM.append(silhouetteBKM)
##



data_KM = {'n_clusters': range(3, 10), 'silhouette': total_silhouette_KM}
df_KM = pd.DataFrame(data_KM)
list(df_KM.columns)


fig = go.Figure(data=[go.Table(header=dict(values=list(df_KM.columns)),
                 cells=dict(values=[df_KM.n_clusters, df_KM.silhouette]))
                     ])
fig.show()

max_silhouette_KM = df_KM[df_KM['silhouette']==df_KM['silhouette'].max()]
n_clusters_KM = int(max_silhouette_KM.iloc[0]['n_clusters'])
print('Number of clusters: ' + str(n_clusters_KM))



data_BKM = {'n_clusters': range(3, 10), 'silhouette': total_silhouette_BKM}
df_BKM = pd.DataFrame(data_BKM)
list(df_BKM.columns)


fig = go.Figure(data=[go.Table(header=dict(values=list(df_BKM.columns)),
                 cells=dict(values=[df_BKM.n_clusters, df_BKM.silhouette]))
                     ])
fig.show()

max_silhouette_BKM = df_BKM[df_BKM['silhouette']==df_BKM['silhouette'].max()]
n_clusters_BKM = int(max_silhouette_BKM.iloc[0]['n_clusters'])
print('Number of clusters: ' + str(n_clusters_BKM))






#cLUSTERING
#model, predictions, silhouette = K_Means(n_clusters_KM, featureDf)
model, predictions, silhouette = BisectingK_Means(n_clusters_BKM, featureDf)
featureDf = predictions.drop("features").drop("symbol").drop("date").drop("label")

#featureDf = dataset.select("*").toPandas()
#featureDf = featureDf.withColumn("FirstName",split(col("features"),",").getItem(0)).withColumn("SName",split(col("features"),",").getItem(1)).withColumn("TName",split(col("features"),",").getItem(2))


print(featureDf.printSchema())
featureDf = featureDf.select("*").toPandas()

perplexity = 30

#T-SNE with two dimensions
tsne_2d = TSNE(n_components=2, perplexity=perplexity)

#T-SNE with three dimensions
tsne_3d = TSNE(n_components=3, perplexity=perplexity)

#This DataFrame contains two dimensions, built by T-SNE
TCs_2d = pd.DataFrame(tsne_2d.fit_transform(featureDf.drop(["prediction"], 1)))
TCs_2d.columns = ["TC1_2d","TC2_2d"]

#And this DataFrame contains three dimensions, built by T-SNE
TCs_3d = pd.DataFrame(tsne_3d.fit_transform(featureDf.drop(["prediction"], axis=1)))
TCs_3d.columns = ["TC1_3d","TC2_3d","TC3_3d"]

print(TCs_2d.head())
#And this DataFrame contains three dimensions, built by T-SNE
#TCs_3d = pd.DataFrame(tsne_3d.fit_transform(featureDf))

featureDf = pd.concat([featureDf,TCs_2d, TCs_3d], axis=1, join='inner')


featureDf["dummy"] = 0

clusters = []
for i in range(0, n_clusters):
    clusters.append(featureDf[featureDf["prediction"] == i])



#Instructions for building the 2-D plot
data = []
for i in range(0, n_clusters):
    data.append(go.Scatter(
                    x = clusters[i]["TC1_2d"],
                    y = clusters[i]["TC2_2d"],
                    mode = "markers",
                    name = "Cluster " + str(i),
                    #marker = dict(color = 'rgba(255, 128, 255, 0.8)'),
                    text = None))



title = "Visualizing Clusters in Two Dimensions Using T-SNE (perplexity=" + str(perplexity) + ")"

layout = dict(title = title,
              xaxis= dict(title= 'TC1',ticklen= 5,zeroline= False),
              yaxis= dict(title= 'TC2',ticklen= 5,zeroline= False)
             )

fig = dict(data = data, layout = layout)

iplot(fig)




#Instructions for building the 3-D plot

data = []
for i in range(0, n_clusters):
    data.append(go.Scatter3d(
                    x = clusters[i]["TC1_3d"],
                    y = clusters[i]["TC2_3d"],
                    z = clusters[i]["TC3_3d"],
                    mode = "markers",
                    name = "Cluster " + str(i),
                    #marker = dict(color = 'rgba(255, 128, 255, 0.8)'),
                    text = None))



title = "Visualizing Clusters in Three Dimensions Using T-SNE (perplexity=" + str(perplexity) + ")"

layout = dict(title = title,
              xaxis= dict(title= 'TC1',ticklen= 5,zeroline= False),
              yaxis= dict(title= 'TC2',ticklen= 5,zeroline= False)
             )

fig = dict(data = data, layout = layout)

iplot(fig)






spark_context.stop()
