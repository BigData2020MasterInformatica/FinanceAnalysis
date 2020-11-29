import sys
from pyspark.sql import SparkSession
from pyspark.sql.functions import window
from pyspark.sql.types import StructType
from pyspark.sql.functions import lit, avg
from pyspark.sql.window import Window
import pyspark.sql.functions as F
from pyspark.ml.clustering import KMeansModel
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import pandas as pd
from pyspark.sql.types import DecimalType


def main(directory) -> None:
    """ Program that reads temperatures in streaming from a directory, finding those that are higher than a given
    threshold.

    It is assumed that an external entity is writing files in that directory, and every file contains a
    temperature value.

    :param directory: streaming directory
    """
    spark = SparkSession \
        .builder \
        .master("local[2]") \
        .appName("StreamingStreaming") \
        .getOrCreate()

    # set log level
    spark.sparkContext.setLogLevel("ERROR")

    # define csv structure
    fileSchema = StructType().add('symbol', 'string').add('date', 'timestamp').add('open', 'string').add('high', 'string').add('low', 'string').add('close', 'string').add('adjClose', 'string') \
    .add('volume', 'string').add('unadjustedVolume', 'string').add('change', 'string').add('changePercent', 'string').add('vwap', 'string').add('label', 'string').add('year', 'string').add('changeOverTime', 'string')

    # Create DataFrame representing the stream of input lines
    lines = spark \
        .readStream \
        .format("CSV") \
        .option('sep', ',') \
        .option("header", "true") \
        .option('includeTimestamp', 'true') \
        .option("timestampFormat", "dd/MM/yyyy") \
        .schema(fileSchema) \
        .load(directory)

    lines.printSchema()
    
    # selects columns of dataframe
    words = lines.select(lines.symbol, lines.volume, lines.date, lines.open, lines.high, lines.low, lines.adjClose, lines.unadjustedVolume, lines.change, lines.changePercent, lines.changeOverTime)
    
    words.printSchema()

    # Generate running, indicating window parameters (in seconds), watermark and calculating aggregations
    windowSize = '{} days'.format(6)
    slideSize = '{} days'.format(4)
    waterMarkSize = '{} days'.format(10)
    windowedCounts = words \
        .withWatermark("date", waterMarkSize) \
        .groupBy(
            window(words.date, windowSize, slideSize),
            words.symbol
        ).agg(F.expr('percentile(volume, array(0.25))')[0].cast(DecimalType()).alias('%25(volume)'),
             F.expr('percentile(volume, array(0.50))')[0].cast(DecimalType()).alias('%50(volume)'),
             F.expr('percentile(volume, array(0.75))')[0].cast(DecimalType()).alias('%75(volume)'),
             F.min(words.volume).alias('min(volume)'),
             F.avg(words.volume).cast(DecimalType()).alias('avg(volume)'),
             F.max(words.volume).alias('max(volume)'),
             F.avg(words.open).cast(DecimalType()).alias('avg(open)'),
             F.avg(words.high).cast(DecimalType()).alias('avg(high)'),
             F.avg(words.low).cast(DecimalType()).alias('avg(low)'),
             F.avg(words.adjClose).cast(DecimalType()).alias('avg(adjClose)'),
             F.avg(words.unadjustedVolume).cast(DecimalType()).alias('avg(unadjustedVolume)'),
             F.avg(words.change).cast(DecimalType()).alias('avg(change)'),
             F.avg(words.changePercent).cast(DecimalType()).alias('avg(changePercent)'),
             F.avg(words.changeOverTime).cast(DecimalType()).alias('avg(changeOverTime)'),
             F.count(words.volume).alias('inputs')) 


    model = KMeansModel.load("D:\gbac\Downloads\modelKM")

    # calculate moving average
    def updateFunction(newValues, batch_id):
        print(batch_id)
        # calcule moving average
        newValues = newValues.withColumn("movingAverage", avg(newValues["avg(volume)"])
             .over( Window.partitionBy("symbol").rowsBetween(-1,1)) )
        
        # create features vector to use kmeans 
        cols = ["avg(high)", "avg(low)", "avg(open)", "avg(adjClose)", "avg(volume)", "avg(unadjustedVolume)", "avg(change)", "avg(changePercent)", "avg(changeOverTime)"]
        assembler = VectorAssembler(inputCols=cols, outputCol="features")
        featureDf = assembler.transform(newValues)

        predictionsDf = model.transform(featureDf)
        predictionsDf.show()
        print(predictionsDf.columns)
        
        # plot trend chart with the new data
        fig, ax = plt.subplots(figsize=(20,10))

        a1 = [i[0][0] for i in predictionsDf.select("window").orderBy('window', ascending=True).collect()]

        a2 = [int(i[0])  for i in predictionsDf.select("avg(volume)").orderBy('window', ascending=True).collect()]
        
        ax.plot(a1, a2)
        
        ax.set(xlabel='Date', ylabel='Money',
               title='Marker trend')
        
        fig.savefig("markerTrend.png")
        plt.show()
        
        fig, ax = plt.subplots(figsize=(20,10))
        # plot cluster chart with the new data
        clusterDf = predictionsDf.groupby(['prediction']).agg(F.count("avg(volume)"))
        print(clusterDf.columns)
#        fig = plt.figure()
        
        countBar = [i[0] for i in clusterDf.select("prediction").collect()]
        predictionBar = [i[0] for i in clusterDf.select("count(avg(volume))").collect()]
        
        ax.bar(countBar,predictionBar, width=0.3)
        ax.set(xlabel='Number of cluster', ylabel='Total elements', title='Streamings with K-means')
        fig.savefig("ClusterTrend.png")
        plt.show()
        
        # print final dataframe
        predictionsDf.show()
        return predictionsDf
                

    # Start running the query that prints the output in the screen and calculating moving average # update
    query = windowedCounts \
        .writeStream \
        .outputMode("update") \
        .format("console") \
        .foreachBatch(updateFunction) \
        .option('truncate', 'false') \
        .start()
        
    query.awaitTermination()
    

if __name__ == '__main__':
   main(sys.argv[1])
