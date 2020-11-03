import sys
import logging
from pyspark.sql import SparkSession
from pyspark.sql.functions import explode, window
from pyspark.sql.types import StructType


def main(directory) -> None:
    """ Program that reads finance in streaming from a directory, finding those that are higher than a given
    threshold.

    It is assumed that an external entity is writing files in that directory.
    The file should have the CSV format with the schema: ["name":string, "money":string, "time":timestamp]
	
    :param directory: streaming directory
    """
    spark = SparkSession \
        .builder \
        .master("local[2]") \
        .appName("StreamingFinanceData") \
        .getOrCreate()

    spark.sparkContext.setLogLevel("ERROR")

    
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


    # Split the lines into words
    words = lines.select(lines.name, lines.money, lines.time)
  
    words.printSchema()
    
    # Generate running word count, indicating window parameters (in seconds)
    windowSize = '{} seconds'.format(3)
    slideSize = '{} seconds'.format(2)
    windowedCounts = words.groupBy(
            window(words.time, windowSize, slideSize),
            words.name
        ).count()\
        .orderBy('window')
    
    windowedCounts.printSchema() 

    # Start running the query that prints the output in the screen
    query = windowedCounts \
        .writeStream \
        .outputMode("complete") \
        .format("console") \
        .option('truncate', 'false') \
        .start()

    query.awaitTermination()


if __name__ == '__main__':
   main(sys.argv[1])
