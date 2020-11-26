import pandas as pd
import matplotlib.pyplot as plt

from fbprophet import Prophet
from pyspark.sql import SparkSession
from pyspark.ml.classification import MultilayerPerceptronClassifier,RandomForestClassifier, GBTClassifier, DecisionTreeClassifier, LinearSVC
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.sql.functions import lead, isnull, when, monotonically_increasing_id, pandas_udf, PandasUDFType, current_date
from pyspark.sql.window import Window
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler
from pyspark.sql.types import *

from finansp.FinancialModeling import getCompany

__INTERVAL_WIDTH = 0
__PERIODS = 0
__SHOW_CHARTS = False
__MAX_ITER = 1
__HIDDEN_LAYERS = [10]
__BLOCK_SIZE = 128
__SEED = 1234
__VAL_SPLIT = 0.2

__schema = StructType([
    StructField("ds", DateType(), True),
    StructField("yhat", FloatType(), True),
    StructField("trend", FloatType(), True),
    StructField("yhat_lower", FloatType(), True),
    StructField("yhat_upper", FloatType(), True),
    StructField("trend_lower", FloatType(), True),
    StructField("trend_upper", FloatType(), True),
    StructField("multiplicative_terms", FloatType(), True),
    StructField("multiplicative_terms_lower", FloatType(), True),
    StructField("multiplicative_terms_upper", FloatType(), True),
    StructField("daily", FloatType(), True),
    StructField("daily_lower", FloatType(), True),   
    StructField("daily_upper", FloatType(), True),
    StructField("weekly", FloatType(), True),
    StructField("weekly_lower", FloatType(), True),
    StructField("weekly_upper", FloatType(), True),
    StructField("yearly", FloatType(), True),
    StructField("yearly_lower", FloatType(), True),
    StructField("yearly_upper", FloatType(), True),
    StructField("additive_terms", FloatType(), True),
    StructField("additive_terms_lower", FloatType(), True),
    StructField("additive_terms_upper", FloatType(), True),
    StructField("company", StringType(), True )
])
__schema_buy = StructType([
    StructField("ds", DateType(), True),
    StructField("yhat", FloatType(), True),
    StructField("trend", FloatType(), True),
    StructField("yhat_lower", FloatType(), True),
    StructField("yhat_upper", FloatType(), True),
    StructField("trend_lower", FloatType(), True),
    StructField("trend_upper", FloatType(), True),
    StructField("multiplicative_terms", FloatType(), True),   
    StructField("multiplicative_terms_lower", FloatType(), True),
    StructField("multiplicative_terms_upper", FloatType(), True),
    StructField("daily", FloatType(), True),
    StructField("daily_lower", FloatType(), True),
    StructField("daily_upper", FloatType(), True),
    StructField("weekly", FloatType(), True),
    StructField("weekly_lower", FloatType(), True),
    StructField("weekly_upper", FloatType(), True),
    StructField("yearly", FloatType(), True),
    StructField("yearly_lower", FloatType(), True),
    StructField("yearly_upper", FloatType(), True),
    StructField("additive_terms", FloatType(), True),
    StructField("additive_terms_lower", FloatType(), True),
    StructField("additive_terms_upper", FloatType(), True),
    StructField("company", StringType(), True ),
    StructField("training_date", DateType(), True ),
    StructField("label", IntegerType(), True ),
    StructField("id", IntegerType(), True ),
    StructField("yhat_next", FloatType(), True ),
    StructField("features", ArrayType(FloatType()), True ),
    StructField("rawPrediction", ArrayType(FloatType()), True ),
    StructField("probability", ArrayType(FloatType()), True ),
    StructField("prediction", FloatType(), True ),
])

def predict( company_list = [], value_to_predict = "open",
 periods_to_predict = 90, interval_width = 0.95, show_charts = False, number_of_days = 2200 ):
    """
        Predict a the specified value, `value_to_predict`, for each company in the
        list `company_list` in the next `periods_to_predict` days, with an `interval_width`
        using the `last_days`. You can show graph by setting `show_charts` to True.

        Values to predict:
            - "open"
            - "close"
            - "high"
            - "low"
            - "volume"
            - "unadjustedVolume"
            - "change"
            - "changePercent"
            - "vwap"
            - "changeOverTime"
    """
    global __INTERVAL_WIDTH, __PERIODS, __SHOW_CHARTS

    # Preparing data
    history = None
    for company in company_list:

        tmp_history = getCompany( company, number_of_days )
        tmp_history["company"] = [ company for i in range( tmp_history.shape[0] )]

        if history is None:

            history = tmp_history.copy()
        else:

            history = history.append( tmp_history )
    
    if history is not None:

        __INTERVAL_WIDTH = interval_width
        __PERIODS = periods_to_predict
        __SHOW_CHARTS = show_charts

        spark = SparkSession \
            .builder \
            .appName("TimeSeries") \
            .config( "spark.driver.extraJavaOptions", "-Dio.netty.tryReflectionSetAccessible=true") \
            .config('spark.sql.execution.arrow.enable', 'true') \
            .getOrCreate()


        df = spark.createDataFrame( history ) \
            .dropna()
        df = df.select(
            df['date'].alias('ds'),
            df[value_to_predict].alias('y'),
            df['company'].cast( StringType() )
        )

        results = (
            df
            .groupBy( 'company' )
            .apply(__forecast_store_item)
            .withColumn( 'training_date', current_date() )
        )

        return results.toPandas()
    else:

        print( "Something was wrong...")
        return None

@pandas_udf(__schema, PandasUDFType.GROUPED_MAP)
def __forecast_store_item( history ):
    
    global __INTERVAL_WIDTH, __PERIODS, __SHOW_CHARTS
    
    # Instantiate the model, configure the parameters
    model = Prophet(
        interval_width=__INTERVAL_WIDTH,
        growth='linear',
        daily_seasonality=True,
        weekly_seasonality=True,
        yearly_seasonality=True
    )

    # Fit the model
    model.fit( history )

    # Configure predictions
    future_pd = model.make_future_dataframe(
        periods=__PERIODS,
        freq='d',
        include_history=True
    )
    
    # Make predictions
    results_pd = model.predict(future_pd)
    results_pd["company"] = history["company"][0]

    if __SHOW_CHARTS:
        model.plot( results_pd, xlabel='date-'+history['company'][0], ylabel='open' )
        plt.show(block=True)
    
    return results_pd

def to_buy_or_not_to_buy( companies=[], max_iter=1, hidden_layers=[10], blockSize=128, seed=1234, val_split=0.2, 
    value_to_predict="close", periods_to_predict=90, interval_width=0.95, show_charts=False, number_of_days=2200,
    classifier_type="mlp", max_depth=5, max_bins=10, min_instances_per_node=1, num_trees=20 ):
    """
        Should you buy or sell? 0 - WAIT, 1 - BUY, 2 - SELL

        This function tells you when to buy and sell, using `predict` function and a classifier
        ( MultiLayer Perceptron (MLP) by default), in the `predictions` column. You can change the number 
        of epochs to train your classifier, `max_iter`, the number of `hidden_layers`, the batch size
        ( `block_size` ) and the amount of data to be used in the validation set (`validation_split`).

        You can also select some Tree Algorithms to predict and change their parameters:\n
            - RandomForest (`rf`)
            - DecisionTree (`dt`)

        You can select one of these values to make the predictions:\n
            - "open"\n
            - "close"\n
            - "high"\n
            - "low"\n
            - "volume"\n
            - "unadjustedVolume"\n
            - "change"\n
            - "changePercent"\n
            - "vwap"\n
            - "changeOverTime"\n
    """
    global __MAX_ITER, __HIDDEN_LAYERS, __BLOCK_SIZE, __SEED, __VAL_SPLIT, \
        __CLASSIFIER_TYPE, __MAX_DEPTH, __MAX_BINS, __MIN_INSTANCES_PER_NODE, __NUM_TREES#, __REG_PARAM

    # Obtain predictions from predict function
    pdf = predict( 
        company_list = companies,
        value_to_predict = value_to_predict,
        periods_to_predict = periods_to_predict,
        interval_width = interval_width,
        show_charts = show_charts,
        number_of_days = number_of_days
    )

    __MAX_ITER = max_iter
    __HIDDEN_LAYERS = hidden_layers
    __BLOCK_SIZE = blockSize
    __SEED = seed
    __VAL_SPLIT = val_split
    __CLASSIFIER_TYPE = classifier_type
    __MAX_DEPTH = max_depth
    __MAX_BINS = max_bins
    __MIN_INSTANCES_PER_NODE = min_instances_per_node
    __NUM_TREES = num_trees
    # __REG_PARAM = reg_param

    spark = SparkSession \
        .builder \
        .appName("TimeSeries") \
        .config( "spark.driver.extraJavaOptions", "-Dio.netty.tryReflectionSetAccessible=true") \
        .config('spark.sql.execution.arrow.enable', 'true') \
        .getOrCreate()

    df = spark.createDataFrame( pdf ) \
        .withColumn("id", monotonically_increasing_id() )
    
    results = (
        df
        .groupBy( 'company' )
        .apply(__buy_or_not_buy_prediction)
    )

    return results.toPandas()

@pandas_udf(__schema_buy, PandasUDFType.GROUPED_MAP)
def __buy_or_not_buy_prediction( history ):
    
    global __MAX_ITER, __HIDDEN_LAYERS, __BLOCK_SIZE, __SEED, __VAL_SPLIT, \
        __CLASSIFIER_TYPE, __MAX_DEPTH, __MAX_BINS, __MIN_INSTANCES_PER_NODE, __NUM_TREES#, __REG_PARAM

    spark = SparkSession \
        .builder \
        .appName("TimeSeries") \
        .config( "spark.driver.extraJavaOptions", "-Dio.netty.tryReflectionSetAccessible=true") \
        .config('spark.sql.execution.arrow.enable', 'true') \
        .getOrCreate()

    df = spark.createDataFrame(history)

    # Get a window
    w = Window().partitionBy().orderBy("id")
    # Lead-shift using window
    df = df.withColumn("yhat_next", lead(df.yhat).over(w))

    # 1 BUY # 0 WAIT # 2 SELL
    df = df.withColumn("label", when(isnull(df.yhat - df.yhat_next), 0).otherwise(
            when( df.yhat < df.yhat_next, 1).otherwise( 2 )
        )).dropna()
    
    # Obtain `features` column
    assembler = VectorAssembler(
        inputCols=[ "yhat", "trend", "yhat_lower", "yhat_upper",
        "trend_lower", "trend_upper", "multiplicative_terms",
        "multiplicative_terms_upper", "multiplicative_terms_lower",
        "daily", "daily_lower", "daily_upper", "weekly",
        "weekly_lower", "weekly_upper", "yearly", "yearly_lower", "yearly_upper",
        "additive_terms", "additive_terms_lower", "additive_terms_upper" ],
        outputCol="features"
    )
    df = assembler.transform( df )

    # Get train and test data
    splits = df.randomSplit( [1 - __VAL_SPLIT, __VAL_SPLIT], __SEED ) 
    train = splits[0]
    test = splits[1]

    # Hidden layers
    layers = [21]
    layers.extend( __HIDDEN_LAYERS )
    layers.append( 3 )

    # Create the trainer and set its parameters
    if __CLASSIFIER_TYPE == "rf": # Random Forest

        trainer = RandomForestClassifier( 
            maxDepth=__MAX_DEPTH, 
            maxBins=__MAX_BINS,
            seed=__SEED,
            minInstancesPerNode=__MIN_INSTANCES_PER_NODE,
            numTrees=__NUM_TREES
        )
    # elif __CLASSIFIER_TYPE == "gbt": # Gradient-Boosted Tree
    #     trainer = GBTClassifier( 
    #         maxDepth=__MAX_DEPTH,
    #         maxBins=__MAX_BINS,
    #         minInstancesPerNode=__MIN_INSTANCES_PER_NODE,
    #         maxIter=__MAX_ITER,
    #         seed=__SEED
    #     )
    elif __CLASSIFIER_TYPE == "dt": # Decision Tree

        trainer = DecisionTreeClassifier(
            maxDepth=__MAX_DEPTH,
            maxBins=__MAX_BINS,
            minInstancesPerNode=__MIN_INSTANCES_PER_NODE,
            seed=__SEED
        )
    # elif __CLASSIFIER_TYPE == "svc": # LinearSVC

    #     trainer = LinearSVC(
    #         maxIter=__MAX_ITER,
    #         regParam=__REG_PARAM
    #     )
    else:

        trainer = MultilayerPerceptronClassifier(
            maxIter=__MAX_ITER,
            layers=layers, 
            blockSize=__BLOCK_SIZE, 
            seed=__SEED
        )
    # Train the model
    model = trainer.fit(train)

    # Compute accuracy on the test set
    result = model.transform(test)

    return result.toPandas()
