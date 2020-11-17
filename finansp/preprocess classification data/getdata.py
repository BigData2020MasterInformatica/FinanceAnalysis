#Benjamin Vega, Iuliana Ilie

#!/usr/bin/env python
import requests
try:
    # For Python 3.0 and later
    from urllib.request import urlopen
except ImportError:
    # Fall back to Python 2's urllib2
    from urllib2 import urlopen

import json


names = ["AAPL"]
for i in names:
    with open("/Users/benjaminvegaherrera/Desktop/BigData/data_file_"+i+".json", "w") as write_file:
    
        url = "https://financialmodelingprep.com"
        search_query_format = "{}/{}&apikey={}"
        key = "c7cf4ab5efb5bcc3d7ba60f6654df7c5"
        url2 = "/api/v3/historical-price-full/"+i+"?from=1990-01-01&to=2020-11-10"
        # Historical Daily Prices with change and volume interval
        query = search_query_format.format( url, url2, key )
        r = requests.get(query)
        #print(r.json())
        jshit = r.json()
        json.dump(jshit, write_file)
