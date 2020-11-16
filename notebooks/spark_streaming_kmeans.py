######################################################
##             Leticia Yepez, Jose Gomez            ##
######################################################


import sched, time

from pyspark import SparkContext
from pyspark.streaming import StreamingContext

from pyspark.mllib.linalg import Vectors
from pyspark.mllib.clustering import StreamingKMeans
import datetime

def main(directory) -> None:
    """ 
        Program that reads market data in streaming from a directory.

        It is assumed that an external entity is writing files in that directory
    
        :param directory: streaming directory
    """
    
    # Create Spark and Spark Streaming contexts with 10s batch interval
    sc = SparkContext(appName='StreamingKMeansClustering')
    sc.setLogLevel("ERROR")
    ssc = StreamingContext(sc, 1)


    # Extracts value and time from training data and
    # turns into a vector
    def parse_training_data(line):
        cells = line.split(',')
        print(line)
        dateCell = datetime.datetime.strptime(cells[2], '%Y-%m-%d %H:%M:%S.%f')
        vec = Vectors.dense([float(cells[1]), float(dateCell.timestamp())])
        return vec
        
    
    # Search for text files in the informed folder.
    # Because it is a Stream, Spark will constantly monitor
    # this directory and any files added to the directory,
    # will be included in the training data. Each file added,
    # is transformed into a new batch in the stream.
    # The map function, will execute the parseTrainingData function to
    # parse the file data into the training stream.
    training_stream = ssc.textFileStream(directory).map(parse_training_data)


    # Initializes the k-means algorithm with streaming to run on data
    # added to the streaming directory.
    # k = 2: Number of clusters into which the dataset will be divided
    # decayFactor = 1.0: All data, from the beginning, is relevant.
    # 0.0: Use only the most recent data.
    # K-means requires the center of random clusters to start the
    # process:
    # 2: Number of centers to be set
    # 1.0 and 0: weight and seed
    model = StreamingKMeans(k=2, decayFactor=1.0).setRandomCenters(2, 1.0, 0)

    # Print initial centers.
    print('Initial centers: ' + str(model.latestModel().centers))

    # Train model
    model.trainOn(training_stream)

    print('Cluster centers before train: ' + str(str(model.latestModel().centers)))

    # start stream
    ssc.start()

    # Schedule the printing of the values of the centers in periodic times
    s = sched.scheduler(time.time, time.sleep)

    # Function that prints the centers recursively, every 10s
    def print_cluster_centers(sc, model):
        print('Cluster centers: ' + str(str(model.latestModel().centers)))
        s.enter(10, 1, print_cluster_centers, (sc, model))

    # The function for printing the clusters (print_cluster_centers) will be
    # executed every 10s with priority 1. This function accepts two
    # arguments, schedule s and model represented by the variable
    # model
    s.enter(10, 1, print_cluster_centers, (s, model))
    s.run()

    # Await termination
    ssc.awaitTermination()


if __name__ == '__main__':
    main(sys.argv[1])