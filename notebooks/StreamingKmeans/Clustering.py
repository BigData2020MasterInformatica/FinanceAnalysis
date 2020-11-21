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


def K_Means(n_clusters, dataset):
    kmeans = KMeans().setK(n_clusters).setSeed(1)
    model = kmeans.fit(dataset)

    predictions = model.transform(dataset)

    evaluator = ClusteringEvaluator()

    silhouette = evaluator.evaluate(predictions)
    print("For n_clusters = " + str(n_clusters) + " Silhouette with squared euclidean distance = " + str(silhouette))

    ctr = []
    centers = model.clusterCenters()
    return centers, model


def BisectingK_Means(nclusters, dataset):
    bkm = BisectingKMeans().setK(n_clusters).setSeed(1)
    model = bkm.fit(dataset)

    predictions = model.transform(dataset)

    evaluator = ClusteringEvaluator()

    silhouette = evaluator.evaluate(predictions)
    print("For n_clusters = " + str(n_clusters) + " Silhouette with squared euclidean distance = " + str(silhouette))

    centers = model.clusterCenters()

    return centers, model


#########################################################################################################################


spark_conf = SparkConf()
spark_context = SparkContext(conf=spark_conf)
spark = SparkSession(spark_context)

path = "C:/Users/nadia/Documents/Universidad/2year/BigData/Session5/"
KMmodelPath = "models/modelKM/"
BKMmodelPath = "models/modelBKM/"
namefile = "datachangeOverTime.txt"

'''if namefile.find(".txt") != -1:
    dataset = spark.read.format("libsvm").load(path + namefile)
else:
    print("Get the proper format for clustering")'''

# dataset = spark.read.format("libsvm").load("C:/Users/nadia/Documents/Universidad/2year/BigData/datachangeOverTime.txt")
dataset = spark.read.format("libsvm").load(path + namefile)



for n_clusters in range(3, 10):
    print("KMeans cluster centers for " + str(n_clusters) + " nclusters")
    ctr = []
    centers, modelKM = K_Means(n_clusters, dataset)

    try:
        modelKM.save(path + KMmodelPath + "modelKM" + str(n_clusters))
    except:
        modelKM.write().overwrite().save(path + KMmodelPath + "modelKM" + str(n_clusters))
    
    for center in centers:
        ctr.append(center)
        print(center)
    print("\n")

print("\n")

for n_clusters in range(3, 10):
    print("Bisecting k-means cluster centers for " + str(n_clusters) + " nclusters")
    centers, modelBKM = BisectingK_Means(n_clusters, dataset)

    try:
        modelBKM.save(path + BKMmodelPath + "modelBKM" + str(n_clusters))
    except:
        modelBKM.write().overwrite().save(path + BKMmodelPath + "modelBKM" + str(n_clusters))
    
    for center in centers:
        print(center)
    print("\n")



spark_context.stop()
