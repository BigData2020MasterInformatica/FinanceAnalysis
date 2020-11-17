#Benjamin Vega, Iuliana Ilie
import json
import csv


with open("/Users/benjaminvegaherrera/Desktop/BigData/data/data_file.json") as file:
       data = json.load(file)

with open("/Users/benjaminvegaherrera/Desktop/BigData/data/data_file_new.txt", "w") as file:
    for item in data['historical']:
        file.write(str(item['close']) + ' 1:' + str(item['high']) + ' 2:'  + str(item['low'])+ ' 3:'  + str(item['open'])+ ' 4:'  + str(item['adjClose'])
            + ' 5:'  + str(item['volume']) + ' 6:'  + str(item['unadjustedVolume']) + ' 7:'  + str(item['change']) + ' 8:'  + str(item['changePercent']) + ' 9:' + str(item['vwap']) 
            + ' 10:'  + str(item['changeOverTime'])+'\n')

with open("/Users/benjaminvegaherrera/Desktop/BigData/data/data_file_new.csv", "w") as file:

    header = 'label,high,low,open,adjClose,volume,unadjustedVolume,change,changePercent,vwap,changeOverTime'
    file.write(header+'\n')
    for item in data['historical']:
        file.write(str(item['close']) + ',' + str(item['high']) + ','  + str(item['low'])+ ','  + str(item['open'])+ ','  + str(item['adjClose'])
            + ','  + str(item['volume']) + ','  + str(item['unadjustedVolume']) + ','  + str(item['change']) + ','  + str(item['changePercent']) + ',' + str(item['vwap']) 
            + ','  + str(item['changeOverTime'])+'\n')