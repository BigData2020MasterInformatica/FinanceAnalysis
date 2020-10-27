from pyspark.sql import SparkSession
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.regression import GBTRegressor

    
if __name__ == "__main__":

    spark_session = SparkSession\
        .builder\
        .appName("Spark Regression")\
        .getOrCreate()

    # Loads data
    dataset = spark_session\
        .read\
        .format("libsvm")\
        .load("classificationDataLibsvm.txt")

    dataset.printSchema()
    dataset.show()
    
    
    
     # Split the data into training and test sets (30% held out for testing)
    (trainingData, testData) = dataset.randomSplit([0.7, 0.3])

    lr = LinearRegression(maxIter=10, regParam=0.3, elasticNetParam=0.8)

    # Fit the model
    lrModel = lr.fit(trainingData)
    
    
    
    # Print the coefficients and intercept for linear regression
    print("Coefficients: %s" % str(lrModel.coefficients))
    print("Intercept: %s" % str(lrModel.intercept))
    
    # Summarize the model over the training set and print out some metrics
    trainingSummary = lrModel.summary
    print("numIterations: %d" % trainingSummary.totalIterations)
    print("objectiveHistory: %s" % str(trainingSummary.objectiveHistory))
    trainingSummary.residuals.show()
    print("RMSE: %f" % trainingSummary.rootMeanSquaredError)
    print("r2: %f" % trainingSummary.r2)
    
    
    lr_predictions = lrModel.transform(testData)
    lr_predictions.select("prediction","label","features").show(5)
    
    lr_evaluator = RegressionEvaluator(predictionCol="prediction", metricName="r2")
    print("R Squared (R2) on test data = %g" % lr_evaluator.evaluate(lr_predictions))
    

    
    print("Using Gradient-boosted tree regression")


    gbt = GBTRegressor(featuresCol = 'features', labelCol = 'label', maxIter=10)
    gbt_model = gbt.fit(trainingData)
    gbt_predictions = gbt_model.transform(testData)
    gbt_predictions.select('prediction', 'label', 'features').show(5)
    lr_evaluator = RegressionEvaluator(predictionCol="prediction", metricName="r2")
    print("R Squared (R2) on test data = %g" % lr_evaluator.evaluate(gbt_predictions))

    spark_session.stop()
    
    
    
    
    
    
   


