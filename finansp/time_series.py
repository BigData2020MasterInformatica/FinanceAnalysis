import pandas as pd
from fbprophet import Prophet
import matplotlib.pyplot as plt 
from pyspark.sql import SparkSession

from FinancialModeling import getCompany

spark = SparkSession \
        .builder \
        .appName("TimeSeries") \
        .config( "spark.driver.extraJavaOptions", "-Dio.netty.tryReflectionSetAccessible=true") \
        .config('spark.sql.execution.arrow.enable', 'true') \
        .getOrCreate()


model_test = Prophet(
    interval_width=0.95,
    growth='linear',
    daily_seasonality=True,
    weekly_seasonality=True,
    yearly_seasonality=True,
    # seasonality_mode='multiplicative'
)

history_pd = getCompany( "AAPL" )
history_pd["company"] = [ "AAPL" for i in range( history_pd.shape[0] ) ]
history_pd_2 = getCompany( "GOOG" )
history_pd_2["company"] = [ "GOOG" for i in range( history_pd_2.shape[0] ) ]

history_pd = history_pd.append(history_pd_2)
# print( history_pd_2 )

history_pd_open = history_pd[['date', 'open']]
history_pd_open = history_pd_open.rename(columns={'date': 'ds', 'open': 'y'})
# print( history_pd_open )

# df_test = history_pd_open.iloc[:-1, :]
# # print(df_test)
# model_test.fit( df_test )

# def stan_init( m ):
#     res = {}
#     for pname in ['k', 'm', 'sigma_obs']:
#         res[pname] = m.params[pname][0][0]
#     for pname in ['delta', 'beta']:
#         res[pname] = m.params[pname][0]
#     return res

# model = Prophet(
#     interval_width=0.95,
#     growth='linear',
#     daily_seasonality=True,
#     weekly_seasonality=True,
#     yearly_seasonality=True,
#     # seasonality_mode='multiplicative'
# )

# model.fit( history_pd_open, init=stan_init( model_test ) )

# future_pd = model.make_future_dataframe(
#     periods=365,
#     freq='d',
#     include_history=True
# )

# forecast_pd = model.predict( future_pd )
# predict_fig = model.plot( forecast_pd, xlabel='date', ylabel='open')
# # components_fig = model.plot_components( forecast_pd )

# from fbprophet.plot import add_changepoints_to_plot
# a = add_changepoints_to_plot( predict_fig.gca(), model, forecast_pd )

# plt.show(block=True)

# print( "SUCCESS" )

from pyspark.sql.functions import pandas_udf, PandasUDFType, sum, max, col, concat, lit
from pyspark.sql.types import *

schema = StructType([
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
 
    ])
@pandas_udf(schema, PandasUDFType.GROUPED_MAP)
def forecast_store_item(history):

    # instantiate the model, configure the parameters
    model = Prophet(
        interval_width=0.95,
        growth='linear',
        daily_seasonality=True,
        weekly_seasonality=True,
        yearly_seasonality=True,
        # seasonality_mode='multiplicative'
    )

    # fit the model
    model.fit(history)

    # configure predictions
    future_pd = model.make_future_dataframe(
        periods=90,
        freq='d',
        include_history=True
    )

    # make predictions
    results_pd = model.predict(future_pd)

    predict_fig = model.plot( results_pd, xlabel='date-'+history['company'][0], ylabel='open' )
    plt.show(block=True)

    # return predictions
    # print( results_pd.info() )
    return results_pd

import math

if history_pd.shape[0] % 2 != 0:
    history_pd = history_pd.iloc[:-1,:]

# history_pd["pairs"] = [ math.floor( i / 100 ) for i in range( history_pd.shape[0] ) ]
# print( history_pd )

df = spark.createDataFrame( history_pd )

# # print( df.count() )

df = df.dropna()

df = df.select(
    df['date'].alias('ds'),#.cast(DateType()).alias('ds'),
    df['open'].alias('y'),#.cast(FloatType()).alias('y'),
    df['company'].cast(StringType())
)

# # print( df.count() )
df.printSchema()

from pyspark.sql.functions import current_date

results = (
    df
    .groupBy( 'company' )
    # .map(lambda d: pd.DataFrame( [d], columns=['ds', 'y']))
    # .reduce( lambda d1, d2: d1.append(d2) )
    # .filter(lambda d: len(d['ds']) >= 2)
    # .reduce(lambda d1, d2: forecast_store_item( d1.append(d2) ) )
    # .map(lambda d: forecast_store_item( d ) )
    # .withColumn('training_date', current_date())
    .apply(forecast_store_item)
    .withColumn('training_date', current_date())
    )

# results = spark.createDataFrame(results, schema)

print( type( results ) )

## REMEMBER: export OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES
print( results.toPandas() )