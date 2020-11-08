import sys
from pyspark.sql import SparkSession
from pyspark.sql.functions import window
from pyspark.sql.types import StructType
from pyspark.sql.functions import lit, avg
from pyspark.sql.window import Window
import pyspark.sql.functions as F


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
        .appName("StreamingFindHighTemperature") \
        .getOrCreate()

    # set log level
    spark.sparkContext.setLogLevel("ERROR")

    # define csv structure
    fileSchema = StructType().add('name', 'string').add('money', 'string').add('time', 'timestamp')

    # Create DataFrame representing the stream of input lines
    lines = spark \
        .readStream \
        .format("CSV") \
        .option('sep', ',') \
        .option("header", "true") \
        .option('includeTimestamp', 'true') \
        .schema(fileSchema) \
        .load(directory)

    lines.printSchema()
    
    # selects columns of dataframe
    words = lines.select(lines.name, lines.money, lines.time)
    
    words.printSchema()

    # Generate running, indicating window parameters (in seconds), watermark and calculating aggregations
    windowSize = '{} seconds'.format(3)
    slideSize = '{} seconds'.format(2)
    waterMarkSize = '{} seconds'.format(10)
    windowedCounts = words \
        .withWatermark("time", waterMarkSize) \
        .groupBy(
            window(words.time, windowSize, slideSize),
            words.name
        ).agg(F.expr('percentile(money, array(0.25))')[0].alias('%25(money)'),
             F.expr('percentile(money, array(0.50))')[0].alias('%50(money)'),
             F.expr('percentile(money, array(0.75))')[0].alias('%75(money)'),
             F.min(words.money).alias('min(money)'),
             F.avg(words.money).alias('avg(money)'),
             F.max(words.money).alias('max(money)'),
             F.count(words.money).alias('inputs')) \
        .orderBy('window')

    # calculate moving average
    def updateFunction(newValues, batch_id):
        print(batch_id)
        newValues = newValues.withColumn("movingAverage", avg(newValues["avg(money)"])
             .over( Window.partitionBy("name").rowsBetween(-1,1)) )
        newValues.show()
        return newValues


    # Start running the query that prints the output in the screen and calculating moving average # update
    query = windowedCounts \
        .writeStream \
        .outputMode("complete") \
        .format("console") \
        .foreachBatch(updateFunction) \
        .option('truncate', 'false') \
        .start()
        
    query.awaitTermination()
    

if __name__ == '__main__':
   main(sys.argv[1])
