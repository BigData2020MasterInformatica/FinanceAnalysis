import requests 
import click
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
from datetime import date
from datetime import timedelta
import numpy as np
from sklearn.datasets import dump_svmlight_file

from pyspark.ml import Pipeline
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.regression import LinearRegression
from pyspark.ml.regression import GBTRegressor
from pyspark.ml.tuning import CrossValidator
from pyspark.ml.tuning import ParamGridBuilder
from pyspark.ml.regression import IsotonicRegression

url = "https://financialmodelingprep.com"
YOUR_API_KEY = "442cb337e1223ef8e61fbca960b35d47" 
query_format = "{}/{}?apikey={}"
search_query_format = "{}/{}&apikey={}"



#Aditional functions

def visualize(df):
    click.echo(df.tail())
    click.echo(df.describe())
    fig, ax = plt.subplots(figsize=(20,20))
    click.echo("Showing data distribution")
    df.hist(ax = ax)
    plt.show()
    click.echo("Looking for outliers")
    plt.subplots(figsize=(20,20))
    df.boxplot()    
    plt.show()
    plt.subplots(figsize=(20,20))
    corrMatrix = df.corr()
    sn.heatmap(corrMatrix, annot=True)
    plt.show()

def dataframeToLibsvm(df,name):
	y = df.close
	df = df.drop(columns=['label', 'date','close'])
	X =  df[np.setdiff1d(df.columns,['features','label'])]
	f = dump_svmlight_file(X,y,'svmlight_'+name+'.txt',zero_based=False,multilabel=False)

#df -> dataset in libsvm format, name-> company name
def LinearRegressionCli(df,name):

    #Split data: training and test
    (trainingData, testData) = df.randomSplit([0.7, 0.3])
    #print(trainingData.select("label").show(10))

    ### Training

    # model
    lr = LinearRegression(featuresCol = 'features', labelCol = 'label')
    
    # grid to find the best model
    paramGrid = ParamGridBuilder() \
    .addGrid(lr.maxIter, [int(x) for x in np.linspace(start = 5, stop = 50, num = 5)]) \
    .addGrid(lr.regParam, [float(x) for x in np.linspace(start = 0.1, stop = 0.9, num = 4)]) \
    .addGrid(lr.elasticNetParam, [float(x) for x in np.linspace(start = 0.01, stop = 1.0, num = 6)]) \
    .build()

    #Evaluator
    evaluator = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="rmse")
    
    # Cross Validation Training
    crossval = CrossValidator(estimator=lr,
                          estimatorParamMaps=paramGrid,
                          evaluator=evaluator,
                          numFolds=3)
    #fit
    cvModel = crossval.fit(trainingData)
    
    ### Evaluation

    #predictions
    predictions = cvModel.transform(testData)

    #Calculate RMSE
    rmse = evaluator.evaluate(predictions)
    
    #Get Best model
    bestModel = cvModel.bestModel
    
    # format: company, model, rmse, predictions
    return {"company":name,"model":bestModel,"rmse":rmse,"predictions":predictions}


#df -> dataset in libsvm format, name-> company name
def GradienBoostingRegressorCli(df,name):

    #Split data: training and test
    (trainingData, testData) = df.randomSplit([0.7, 0.3])
    #print(trainingData.select("label").show(10))

    ### Training

    #model
    gbt = GBTRegressor(featuresCol = 'features', labelCol = 'label')
    
    # grid to find the best model
    paramGrid = ParamGridBuilder() \
    .addGrid(gbt.maxIter, [int(x) for x in [5,10,30]]) \
    .addGrid(gbt.maxDepth, [int(x) for x in [10]]) \
    .addGrid(gbt.maxBins, [int(x) for x in [32]]) \
    .build()

    #Evaluator
    evaluator = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="rmse")

    # Cross Validation Training
    crossval = CrossValidator(estimator=gbt,
                          estimatorParamMaps=paramGrid,
                          evaluator=evaluator,
                          numFolds=3)
    
    #fit
    cvModel = crossval.fit(trainingData)
    
    ### Evaluation

    #Predictions
    predictions = cvModel.transform(testData)
    
    #Calculate RMSE
    rmse = evaluator.evaluate(predictions)
    
    #Get Best model
    bestModel = cvModel.bestModel
    
    # format: company, model, rmse, predictions
    return {"company":name,"model":bestModel,"rmse":rmse,"predictions":predictions}


#df -> dataset in libsvm format, name-> company name
def IsotonicRegressionCli(df,name):
    
    #Split data: training and test
    (trainingData, testData) = df.randomSplit([0.7, 0.3])

    ### Training
    
    #Model
    ir = IsotonicRegression(featuresCol = 'features', labelCol = 'label')
   
    
    # grid to find the best model
    paramGrid = ParamGridBuilder() \
    .addGrid(ir.isotonic, [x for x in [True,False]]) \
    .build()

    #Evaluator
    evaluator = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="rmse")
    
    # Cross Validation Training
    crossval = CrossValidator(estimator=ir,
                          estimatorParamMaps=paramGrid,
                          evaluator=evaluator,
                          numFolds=3)
    
    #fit
    cvModel = crossval.fit(trainingData)
    
    ### Evaluation

    #prediction
    predictions = cvModel.transform(testData)
    
    #Calculate RMSE
    rmse = evaluator.evaluate(predictions)
    
    #Get Best model
    bestModel = cvModel.bestModel
    
    # format: company, model, rmse, predictions
    return {"company":name,"model":bestModel,"rmse":rmse,"predictions":predictions}




#CLI Implementation
@click.group()
def cli():
    pass

#Get the essential information about the company from ticket
@click.command(help = "Get the company information from ticket")
@click.option('--ticket', prompt = "Insert the ticket company: ")
def getCompanyInfo(ticket):
    query = query_format.format( url, "/api/v3/profile/" + str(ticket), YOUR_API_KEY )
    r = requests.get(query)
    data = r.json()

    company_inf = {
                 "ticker": data[0]["symbol"],
                 "name": data[0]["companyName"],
                 "sector" : data[0]["sector"],
                 "price" : data[0]["price"],
                 "country": data[0]["country"],
                 "state" : data[0]["state"],
                 "city" : data[0]["city"],
                 "address" : data[0]["address"],
                 "website" : data[0]["website"]
                }
  
    click.echo(company_inf)   

#Get the information about the company from ticket for a certain period of time
@click.command(help = "Get history of a company by ticket and period")
@click.option('--ticket', prompt = "Insert the ticket company: ")
@click.option('--start', prompt = "Insert the starting date (AAAA-MM-DD): ")
@click.option('--end', prompt = "Insert the ending date (AAAA-MM-DD): ")
def getCompanyHistory(ticket,start,end):
    query = search_query_format.format( url, "/api/v3/historical-price-full/"+str(ticket)+"?from="+str(start)+"&to="+str(end), YOUR_API_KEY )
    r = requests.get(query)
    data = r.json()
    if data:
        df = pd.DataFrame.from_dict(data["historical"])
        visualize(df)
    else:
        click.echo("No information of this company")

#Get the information about the company from ticket for last week
@click.command(help = "Get last seven days of a company by ticket and period")
@click.option('--ticket', prompt = "Insert the ticket company: ")
def getCompanyLastSevenDays(ticket):
    today = date.today()
    starting_date = today - timedelta(days=7)
    start = starting_date.strftime("%Y-%m-%d")
    end = today.strftime("%Y-%m-%d")
    query = search_query_format.format( url, "/api/v3/historical-price-full/"+str(ticket)+"?from="+str(start)+"&to="+str(end), YOUR_API_KEY )
    r = requests.get(query)
    data = r.json()
    click.echo(data)
    if data:
        df = pd.DataFrame.from_dict(data["historical"])
        visualize(df)
    else:
        click.echo("No information of this company")

def getCompany(ticket):
    today = date.today()
    starting_date = today - timedelta(days=1825)
    start = starting_date.strftime("%Y-%m-%d")
    end = today.strftime("%Y-%m-%d")
    query = search_query_format.format( url, "/api/v3/historical-price-full/"+str(ticket)+"?from="+str(start)+"&to="+str(end), YOUR_API_KEY )
    r = requests.get(query)
    data = r.json()
    if data:
        df = pd.DataFrame.from_dict(data["historical"])
    else:
        print("No information of this company")
    return df

cli.add_command(getCompanyInfo)
cli.add_command(getCompanyHistory)
cli.add_command(getCompanyLastSevenDays)



if __name__ == '__main__':
    cli()
    
