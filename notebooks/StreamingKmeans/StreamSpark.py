'''
Jose Gómez Baco
Leticia Yepez Chavez
Nadia Carrera Chahir
'''
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


def main(directory) -> None:
    """ Program that reads marker values in streaming from a directory.

    It is assumed that an external entity is writing files in that directory. Load a K-Means model to  predict the cluster.

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
    .add('volume', 'string').add('unadjustedVolume', 'string').add('change', 'string').add('changePercent', 'string').add('vwap', 'string').add('label', 'string').add('changeOverTime', 'string')

    # Create DataFrame representing the stream of input lines
    lines = spark \
        .readStream \
        .format("CSV") \
        .option('sep', ',') \
        .option("header", "true") \
        .option('includeTimestamp', 'true') \
        .option("timestampFormat", "yyyy-MM-dd") \
        .schema(fileSchema) \
        .load(directory)

    lines.printSchema()
    
    # selects columns of dataframe
    words = lines.select(lines.symbol, lines.volume, lines.date, lines.open, lines.high, lines.low, lines.adjClose, lines.unadjustedVolume, lines.change, lines.changePercent, lines.changeOverTime)
    
    words.printSchema()

    # Generate running, indicating window parameters (in seconds), watermark and calculating aggregations
    windowSize = '{} days'.format(3)
    slideSize = '{} days'.format(2)
    waterMarkSize = '{} days'.format(10)
    windowedCounts = words \
        .withWatermark("date", waterMarkSize) \
        .groupBy(
            window(words.date, windowSize, slideSize),
            words.symbol
        ).agg(F.expr('percentile(volume, array(0.25))')[0].alias('%25(volume)'),
             F.expr('percentile(volume, array(0.50))')[0].alias('%50(volume)'),
             F.expr('percentile(volume, array(0.75))')[0].alias('%75(volume)'),
             F.min(words.volume).alias('min(volume)'),
             F.avg(words.volume).alias('avg(volume)'),
             F.max(words.volume).alias('max(volume)'),
             F.avg(words.open).alias('avg(open)'),
             F.avg(words.high).alias('avg(high)'),
             F.avg(words.low).alias('avg(low)'),
             F.avg(words.adjClose).alias('avg(adjClose)'),
             F.avg(words.unadjustedVolume).alias('avg(unadjustedVolume)'),
             F.avg(words.change).alias('avg(change)'),
             F.avg(words.changePercent).alias('avg(changePercent)'),
             F.avg(words.changeOverTime).alias('avg(changeOverTime)'),
             F.count(words.volume).alias('inputs')) 


    model = KMeansModel.load("D:\gbac\Downloads\modelKM9\modelKM9")

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
        fig, ax = plt.subplots()
        plt.figure(figsize=(20,20))

        a1 = [i[0] for i in predictionsDf.select("window").orderBy('window', ascending=True).collect()]

        a2 = [i[0] for i in predictionsDf.select("avg(volume)").orderBy('window', ascending=True).collect()]
        ax.plot(a1, a2)
        
        ax.set(xlabel='Date', ylabel='Money',
               title='Marker trend')
        
        fig.savefig("markerTrend.png")
        plt.show()
        
        # plot cluster chart with the new data
        clusterDf = predictionsDf.groupby(['prediction']).agg(F.count("avg(volume)"))
        print(clusterDf.columns)
        fig = plt.figure()
        ax = fig.add_axes([0,0,1,1])
        
        countBar = [i[0] for i in clusterDf.select("prediction").collect()]
        predictionBar = [i[0] for i in clusterDf.select("count(avg(volume))").collect()]
        
        ax.bar(countBar,predictionBar)
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
