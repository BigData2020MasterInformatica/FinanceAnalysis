import pandas as pd
from fbprophet import Prophet
import matplotlib.pyplot as plt
from pyspark.sql import SparkSession
from pyspark.sql.functions import current_date
from pyspark.sql.functions import pandas_udf, PandasUDFType, sum, max, col, concat, lit
from pyspark.sql.types import *

from finansp.FinancialModeling import getCompany

__INTERVAL_WIDTH = 0
__PERIODS = 0
__SHOW_CHARTS = False

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
            - "close"
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


__schema = StructType([
        # 0   ds                          92 non-null     datetime64[ns]
        StructField("ds", DateType(), True),
        # 18  yhat                        92 non-null     float64 
        StructField("yhat", FloatType(), True),
        # 1   trend                       92 non-null     float64   
        StructField("trend", FloatType(), True),
        # 2   yhat_lower                  92 non-null     float64       
        StructField("yhat_lower", FloatType(), True),
        # 3   yhat_upper                  92 non-null     float64       
        StructField("yhat_upper", FloatType(), True),
        # 4   trend_lower                 92 non-null     float64       
        StructField("trend_lower", FloatType(), True),
        # 5   trend_upper                 92 non-null     float64       
        StructField("trend_upper", FloatType(), True),
        # 6   multiplicative_terms        92 non-null     float64       
        StructField("multiplicative_terms", FloatType(), True),
        # 7   multiplicative_terms_lower  92 non-null     float64       
        StructField("multiplicative_terms_lower", FloatType(), True),
        # 8   multiplicative_terms_upper  92 non-null     float64       
        StructField("multiplicative_terms_upper", FloatType(), True),
        StructField("daily", FloatType(), True),
        # 10  weekly_lower                92 non-null     float64       
        StructField("daily_lower", FloatType(), True),
        # 11  weekly_upper                92 non-null     float64       
        StructField("daily_upper", FloatType(), True),
        # 9   weekly                      92 non-null     float64       
        StructField("weekly", FloatType(), True),
        # 10  weekly_lower                92 non-null     float64       
        StructField("weekly_lower", FloatType(), True),
        # 11  weekly_upper                92 non-null     float64       
        StructField("weekly_upper", FloatType(), True),
        # 12  yearly                      92 non-null     float64       
        StructField("yearly", FloatType(), True),
        # 13  yearly_lower                92 non-null     float64       
        StructField("yearly_lower", FloatType(), True),
        # 14  yearly_upper                92 non-null     float64       
        StructField("yearly_upper", FloatType(), True),
        # 15  additive_terms              92 non-null     float64       
        StructField("additive_terms", FloatType(), True),
        # 16  additive_terms_lower        92 non-null     float64       
        StructField("additive_terms_lower", FloatType(), True),
        # 17  additive_terms_upper        92 non-null     float64       
        StructField("additive_terms_upper", FloatType(), True),
        StructField("company", StringType(), True )
    ])
@pandas_udf(__schema, PandasUDFType.GROUPED_MAP)
def __forecast_store_item( history ):
    
    global __INTERVAL_WIDTH, __PERIODS, __SHOW_CHARTS
    
    # instantiate the model, configure the parameters
    model = Prophet(
        interval_width=__INTERVAL_WIDTH,
        growth='linear',
        daily_seasonality=True,
        weekly_seasonality=True,
        yearly_seasonality=True
    )

    # fit the model
    model.fit( history )

    # configure predictions
    future_pd = model.make_future_dataframe(
        periods=__PERIODS,
        freq='d',
        include_history=True
    )
    
    # make predictions
    results_pd = model.predict(future_pd)
    results_pd["company"] = history["company"][0]

    if __SHOW_CHARTS:
        model.plot( results_pd, xlabel='date-'+history['company'][0], ylabel='open' )
        plt.show(block=True)
    
    return results_pd