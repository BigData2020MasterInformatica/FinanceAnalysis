import requests 
import click
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
from datetime import date
from datetime import timedelta

url = "https://financialmodelingprep.com"
YOUR_API_KEY = "442cb337e1223ef8e61fbca960b35d47" 
query_format = "{}/{}?apikey={}"
search_query_format = "{}/{}&apikey={}"

#Aditional functions
def visualize(df):
    click.echo(df.tail())
    click.echo(df.describe())
    input("Press any key to continue")
    click.echo("Showing data distribution")
    df.hist()
    plt.show()
    input("Press any key to continue")
    click.echo("Looking for outliers")
    df.boxplot()    
    plt.show()
    input("Press any key to continue")
    click.echo("Showing correlation between attributes")
    corrMatrix = df.corr()
    sn.heatmap(corrMatrix, annot=True)
    plt.show()

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

def getCompany(ticket, number_of_days):
    today = date.today()
    starting_date = today - timedelta(days=number_of_days)
    start = starting_date.strftime("%Y-%m-%d")
    end = today.strftime("%Y-%m-%d")
    query = search_query_format.format( url, "/api/v3/historical-price-full/"+str(ticket)+"?from="+str(start)+"&to="+str(end), YOUR_API_KEY )
    r = requests.get(query)
    data = r.json()
    # click.echo(data)
    if data:
        df = pd.DataFrame.from_dict(data["historical"])
        # visualize(df)
    else:
        click.echo("No information of this company")
    # print( "hi" )
    return df

cli.add_command(getCompanyInfo)
cli.add_command(getCompanyHistory)
cli.add_command(getCompanyLastSevenDays)



if __name__ == '__main__':
    cli()


