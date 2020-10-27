# Required libraries
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler
from pyspark.context import SparkContext
from pyspark.sql.session import SparkSession
from pyspark.ml.clustering import KMeans, BisectingKMeans
from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark import SparkConf, SparkContext

### Libraries for data
import pandas_datareader.data as web
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import datetime




spark_conf = SparkConf()
spark_context = SparkContext(conf=spark_conf)
spark = SparkSession(spark_context)

dataset = spark.read.format("libsvm").load("data.txt")



# Trains a k-means model.
print("k-means model")
n_clusters = 3
kmeans = KMeans().setK(n_clusters).setSeed(1)
model = kmeans.fit(dataset)

# Make predictions
predictions = model.transform(dataset)

# Evaluate clustering by computing Silhouette score
evaluator = ClusteringEvaluator()

silhouette = evaluator.evaluate(predictions)
print("[k-means] Silhouette with squared euclidean distance = " + str(silhouette))


print("[k-means] Cluster Centers: ")
ctr=[]
centers = model.clusterCenters()
for center in centers:
    ctr.append(center)
    print(center)




# Trains a bisecting k-means model.
print("Bisecting k-means model")
bkm = BisectingKMeans().setK(n_clusters).setSeed(1)
model = bkm.fit(dataset)

# Make predictions
predictions = model.transform(dataset)

# Evaluate clustering by computing Silhouette score
evaluator = ClusteringEvaluator()

silhouette = evaluator.evaluate(predictions)
print("[bisecting k-means] Silhouette with squared euclidean distance = " + str(silhouette))


print("[bisecting k-means] Cluster Centers: ")
centers = model.clusterCenters()
for center in centers:
    print(center)

spark_context.stop()
