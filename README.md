# Real Estate Acquisition Advisor V.2

The Real Estate Acquisition Advisor V.2 is a comprehensive tool that offers a multifaceted approach to assist users in making informed investment decisions. It encompasses various crucial aspects of real estate investment, starting from data retrieval and cleaning, all the way to machine learning-driven rent prediction models and in-depth financial analysis. This tool leverages web scraping to gather property listings from popular platforms like Redfin and Zillow, ensuring access to up-to-date and relevant data. It also provides user inputs for interactive cash flow projection and visualization, allowing users to evaluate the cash flow potential of different properties based on their downpayment preferences and location-specific rental comparables. Furthermore, it incorporates machine learning models to predict property rents, taking into account property square footage, which adds an extra layer of accuracy to rent projections. Finally, it goes beyond the basics by offering extensive financial insights, including property value projections, MOIC calculations, neighborhood analytics, and property value trends by zip code. Overall, this tool equips users with a holistic toolkit to make well-informed real estate investment decisions.


```python
#MUST BE RAN
import numpy as np
import scipy as sp
import os
import pandas as pd
import glob
import os.path
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.image as mpimg
import seaborn as sns                       # pandas for data management
import geopandas                            # geopandas for maps work
from shapely.geometry import Point # shapely handles the coordinate references for plotting shapes
from cartopy import crs as ccrss
import contextily as ctx
import folium
import pyproj
from shapely.geometry import shape
import numpy_financial as npf
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.linear_model import BayesianRidge
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
import itertools
import time
import statsmodels.api as sm
from PIL import Image


#main computer
os.chdir(r"C:\Users\Administrator\Dropbox\Apps\ScrapeHero-Cloud") #main computer

```

# Data Retrieval, Cleaning, and Preparations

In the following Python code blocks, a series of essential tasks are performed, encompassing data retrieval, cleaning, and preparation. Initially, data is automatically obtained through web scraping and stored in a Dropbox folder for ease of access. Subsequently, the data is read into dataframes and undergoes necessary cleaning and formatting. The integration with Dropbox enables seamless data usage across various computers. The dataset consists of property listings from Redfin and Zillow, encompassing properties for sale on both platforms and Zillow rentals. For expense calculations there is a primary focus on Redfin data due to its proven accuracy in expense projections. Monthly expenses are categorized into 5% and 20% downpayment scenarios to account for the significant impact of PMI on margins, a crucial consideration for later analyses.


```python
#MUST BE RAN
#REDFIN BUY SCRAPE
#Grabs most recent uploaded file
#Scraper 1
folder_path3 = r'C:\Users\Administrator\Dropbox\Apps\ScrapeHero-Cloud\RedfinBuyScrape'
file_type3 = r'\*csv'
files3 = glob.glob(folder_path3 + file_type3)
max_file3 = max(files3, key=os.path.getctime)

#Scraper2
folder_path31 = r'C:\Users\Administrator\Dropbox\Apps\ScrapeHero-Cloud\RedfinBuyScrape2'
file_type31 = r'\*csv'
files31 = glob.glob(folder_path31 + file_type31)
max_file31 = max(files31, key=os.path.getctime)

#Cleaning data
RedfinBuyScrape1 = pd.read_csv(max_file3)
RedfinBuyScrape2 = pd.read_csv(max_file31)
RedfinBuyScrape = pd.concat([RedfinBuyScrape1, RedfinBuyScrape2], ignore_index=True)
del RedfinBuyScrape['rank']
del RedfinBuyScrape['property_id']
del RedfinBuyScrape['currency']
del RedfinBuyScrape['land_area_sqft']
del RedfinBuyScrape['property_type']
del RedfinBuyScrape['property_style']
del RedfinBuyScrape['sold_date']
del RedfinBuyScrape['listing_agent_name']
del RedfinBuyScrape['listing_broker_name']
del RedfinBuyScrape['other_agents']
del RedfinBuyScrape['image_urls']
del RedfinBuyScrape['year_built']
del RedfinBuyScrape['status']
del RedfinBuyScrape['mls_number']
del RedfinBuyScrape['input_url']
del RedfinBuyScrape['description']
del RedfinBuyScrape['listing_page_url']
del RedfinBuyScrape['redfin_estimate']
#Adjusts mortgage cost to estimate for .05 down rather than .2
RedfinBuyScrape['estimate_mortgage'] = RedfinBuyScrape['estimate_mortgage'] + (RedfinBuyScrape['price']*.0015388)
RedfinBuyScrape['bedrooms'] = RedfinBuyScrape['bedrooms'].astype('float64')





print(RedfinBuyScrape)
RedfinBuyScrape.dtypes
```

                                              address   latitude  longitude  \
    0     208 W Washington St #709, Chicago, IL 60606  41.883391 -87.634297   
    1       500 W Superior St #801, Chicago, IL 60654  41.895951 -87.642196   
    2        400 W Ontario St #505, Chicago, IL 60654  41.893401 -87.638723   
    3      333 N Jefferson St #303, Chicago, IL 60661  41.887826 -87.642396   
    4      500 W Superior St #1411, Chicago, IL 60654  41.895951 -87.642196   
    ..                                            ...        ...        ...   
    67       834 N Wood St Unit 3S, Chicago, IL 60622  41.897055 -87.672342   
    68  714 W Evergreen Ave Unit A, Chicago, IL 60610  41.906571 -87.646763   
    69     21 W Goethe St Unit 8DE, Chicago, IL 60610  41.905459 -87.629626   
    70  1009 N Oakley Blvd Unit 1W, Chicago, IL 60622  41.899736 -87.684377   
    71      401 E Ontario St #1609, Chicago, IL 60611  41.893211 -87.616401   
    
         price  bedrooms  bathrooms  area_sqft  estimate_mortgage days_on_market  \
    0   319000       2.0        1.0      850.0         3220.87720       15 hours   
    1   560000       2.0        2.5        NaN         6188.72800         6 days   
    2   374900       2.0        2.0     1100.0         3906.89612         7 days   
    3   479000       2.0        2.0     1400.0         4715.08520          1 day   
    4   550000       2.0        2.0     1265.0         5744.34000         6 days   
    ..     ...       ...        ...        ...                ...            ...   
    67  409900       2.0        1.5     1200.0         3788.75412         6 days   
    68  475000       2.0        1.5     1260.0         4452.93000         6 days   
    69  379900       2.0        2.0        NaN         4542.59012       33 hours   
    70  280000       2.0        1.0        NaN         2652.86400         5 days   
    71  425000       2.0        2.0     1200.0         4926.99000         2 days   
    
                                             property_url  
    0   https://www.redfin.com/IL/Chicago/208-W-Washin...  
    1   https://www.redfin.com/IL/Chicago/500-W-Superi...  
    2   https://www.redfin.com/IL/Chicago/400-W-Ontari...  
    3   https://www.redfin.com/IL/Chicago/333-N-Jeffer...  
    4   https://www.redfin.com/IL/Chicago/500-W-Superi...  
    ..                                                ...  
    67  https://www.redfin.com/IL/Chicago/834-N-Wood-S...  
    68  https://www.redfin.com/IL/Chicago/714-W-Evergr...  
    69  https://www.redfin.com/IL/Chicago/21-W-Goethe-...  
    70  https://www.redfin.com/IL/Chicago/1009-N-Oakle...  
    71  https://www.redfin.com/IL/Chicago/401-E-Ontari...  
    
    [72 rows x 10 columns]
    




    address               object
    latitude             float64
    longitude            float64
    price                  int64
    bedrooms             float64
    bathrooms            float64
    area_sqft            float64
    estimate_mortgage    float64
    days_on_market        object
    property_url          object
    dtype: object




```python
#MUST BE RAN
#ZILLOW RENT SCRAPE
#Grabs most recent uploaded file
#Scaper1
folder_path2 = r'C:\Users\Administrator\Dropbox\Apps\ScrapeHero-Cloud\ZillowRentScrape'
file_type2 = r'\*csv'
files2 = glob.glob(folder_path2 + file_type2)
max_file2 = max(files2, key=os.path.getctime)

#Scraper2
folder_path21 = r'C:\Users\Administrator\Dropbox\Apps\ScrapeHero-Cloud\ZillowRentScrape2'
file_type21 = r'\*csv'
files21 = glob.glob(folder_path21 + file_type21)
max_file21 = max(files21, key=os.path.getctime)

#cleaning data
ZillowRentScrape1 = pd.read_csv(max_file2)
ZillowRentScrape2 = pd.read_csv(max_file21)
ZillowRentScrape = pd.concat([ZillowRentScrape1, ZillowRentScrape2], ignore_index=True)
del ZillowRentScrape['rank']
del ZillowRentScrape['property_id']
del ZillowRentScrape['broker_name']
del ZillowRentScrape['input']
del ZillowRentScrape['listing_url']
del ZillowRentScrape['listing_type']
del ZillowRentScrape['image']
del ZillowRentScrape['zestimate']
del ZillowRentScrape['rent_zestimate']
del ZillowRentScrape['days_on_zillow']
del ZillowRentScrape['sold_date']
del ZillowRentScrape['currency']
del ZillowRentScrape['land_area']
del ZillowRentScrape['is_zillow_owned']
#FOR ML PORTION################################
#ZillowRentScrape['area'] = ZillowRentScrape['area'].str.replace('sqft', '')
#ZillowRentScrape['area'] = ZillowRentScrape['area'].astype('float64')
#ZillowRentScrape['area'].fillna(ZillowRentScrape['area'].mean(), inplace=True)
###################################################
ZillowRentScrape['price'] = ZillowRentScrape['price'].str.replace('+', '.0')
ZillowRentScrape['price'] = ZillowRentScrape['price'].str.replace(',', '')
ZillowRentScrape['price'] = ZillowRentScrape['price'].astype('float64')


print(ZillowRentScrape)
ZillowRentScrape.dtypes
```

                                                 address   latitude  longitude  \
    0      653 N Kingsbury St APT 801, Chicago, IL 60654  41.893540  -87.64122   
    1                        637 N Wells St, Chicago, IL  41.893540  -87.63389   
    2        333 W Hubbard St APT 808, Chicago, IL 60654  41.889545  -87.63613   
    3           363 W Grand Ave #1405, Chicago, IL 60654  41.891544  -87.63786   
    4                     320 W Illinois St, Chicago, IL  41.890926  -87.63665   
    ...                                              ...        ...        ...   
    1030       940 N Damen Ave APT 3R, Chicago, IL 60622  41.898987  -87.67745   
    1031  847 N Winchester Ave APT 2F, Chicago, IL 60622  41.897280  -87.67552   
    1032           1535 N Rockwell St, Chicago, IL 60622  41.908836  -87.69176   
    1033               2541 N Southport Ave, Chicago, IL  41.928260  -87.66307   
    1034       1244 W Schubert Ave #2, Chicago, IL 60614  41.930870  -87.66038   
    
           price  bathrooms  bedrooms         area  \
    0     3600.0        2.0       2.0  1200.0 sqft   
    1     3463.0        NaN       2.0          NaN   
    2     3850.0        2.0       2.0  1425.0 sqft   
    3     4045.0        2.0       2.0  1075.0 sqft   
    4     3310.0        NaN       2.0          NaN   
    ...      ...        ...       ...          ...   
    1030  2185.0        1.0       2.0   900.0 sqft   
    1031  2200.0        1.0       2.0  1000.0 sqft   
    1032  2200.0        1.0       2.0  1100.0 sqft   
    1033  4275.0        NaN       3.0          NaN   
    1034  2000.0        1.0       2.0          NaN   
    
                                               property_url  
    0     https://www.zillow.com/homedetails/653-N-Kings...  
    1     https://www.zillow.com/b/the-gallery-on-wells-...  
    2     https://www.zillow.com/homedetails/333-W-Hubba...  
    3     https://www.zillow.com/homedetails/363-W-Grand...  
    4     https://www.zillow.com/b/river-north-park-apar...  
    ...                                                 ...  
    1030  https://www.zillow.com/homedetails/940-N-Damen...  
    1031  https://www.zillow.com/homedetails/847-N-Winch...  
    1032  https://www.zillow.com/homedetails/1535-N-Rock...  
    1033  https://www.zillow.com/b/2541-n-southport-ave-...  
    1034  https://www.zillow.com/homedetails/1244-W-Schu...  
    
    [1035 rows x 8 columns]
    

    C:\Users\Administrator\AppData\Local\Temp\ipykernel_15384\2363355487.py:39: FutureWarning: The default value of regex will change from True to False in a future version. In addition, single character regular expressions will *not* be treated as literal strings when regex=True.
      ZillowRentScrape['price'] = ZillowRentScrape['price'].str.replace('+', '.0')
    




    address          object
    latitude        float64
    longitude       float64
    price           float64
    bathrooms       float64
    bedrooms        float64
    area             object
    property_url     object
    dtype: object




```python
#MUST BE RAN
#ZILLOW BUY SCRAPE
#grabs most recently uploaded file
folder_path1 = r'C:\Users\Administrator\Dropbox\Apps\ScrapeHero-Cloud\ZillowBuyScrape'
file_type1 = r'\*csv'
files1 = glob.glob(folder_path1 + file_type1)
max_file1 = max(files1, key=os.path.getctime)

#Scraper2
folder_path11 = r'C:\Users\Administrator\Dropbox\Apps\ScrapeHero-Cloud\ZillowBuyScrape2'
file_type11 = r'\*csv'
files11 = glob.glob(folder_path11 + file_type11)
max_file11 = max(files11, key=os.path.getctime)

#Cleaning data
ZillowBuyScrape1 = pd.read_csv(max_file1) 
ZillowBuyScrape2 = pd.read_csv(max_file11) 
ZillowBuyScrape = pd.concat([ZillowBuyScrape1, ZillowBuyScrape2], ignore_index=True)
del ZillowBuyScrape['rank']
del ZillowBuyScrape['property_id']
del ZillowBuyScrape['broker_name']
del ZillowBuyScrape['input']
del ZillowBuyScrape['listing_url']
del ZillowBuyScrape['listing_type']
del ZillowBuyScrape['image']
del ZillowBuyScrape['zestimate']
del ZillowBuyScrape['rent_zestimate']
del ZillowBuyScrape['days_on_zillow']
del ZillowBuyScrape['sold_date']
del ZillowBuyScrape['currency']
del ZillowBuyScrape['land_area']
del ZillowBuyScrape['is_zillow_owned']
ZillowBuyScrape['address'] = ZillowBuyScrape['address'].str.replace('APT ', '#').str.replace('UNIT ', '#')
print(ZillowBuyScrape)
```

                                              address   latitude  longitude  \
    0        200 W Grand Ave #1804, Chicago, IL 60654  41.892040 -87.634476   
    1      700 N Larrabee St #1715, Chicago, IL 60654  41.894608 -87.643090   
    2    208 W Washington St #1510, Chicago, IL 60606  41.883335 -87.634190   
    3     330 S Michigan Ave #1809, Chicago, IL 60604  41.877884 -87.624410   
    4      333 N Jefferson St #303, Chicago, IL 60661  41.887800 -87.642410   
    ..                                            ...        ...        ...   
    118  1735 W Diversey Pkwy #311, Chicago, IL 60614  41.931890 -87.672690   
    119           1881 N Poe St #I, Chicago, IL 60614  41.916195 -87.654140   
    120  1212 N Lake Shore Dr #8CN, Chicago, IL 60610  41.904552 -87.625496   
    121     714 W Evergreen Ave #A, Chicago, IL 60610  41.907238 -87.647660   
    122     849 N Franklin St #404, Chicago, IL 60610  41.897804 -87.635430   
    
            price  bathrooms  bedrooms         area  \
    0    435000.0        2.0       2.0  1127.0 sqft   
    1    495000.0        2.0       2.0  1250.0 sqft   
    2    335000.0        2.0       2.0  1300.0 sqft   
    3    650000.0        3.0       3.0     0.0 sqft   
    4    479000.0        2.0       2.0  1400.0 sqft   
    ..        ...        ...       ...          ...   
    118  399900.0        2.0       2.0  1150.0 sqft   
    119  635000.0        2.0       3.0  1800.0 sqft   
    120  649000.0        2.0       2.0  1850.0 sqft   
    121  475000.0        2.0       2.0  1260.0 sqft   
    122  549900.0        2.0       2.0  1475.0 sqft   
    
                                              property_url  
    0    https://www.zillow.com/homedetails/200-W-Grand...  
    1    https://www.zillow.com/homedetails/700-N-Larra...  
    2    https://www.zillow.com/homedetails/208-W-Washi...  
    3    https://www.zillow.com/homedetails/330-S-Michi...  
    4    https://www.zillow.com/homedetails/333-N-Jeffe...  
    ..                                                 ...  
    118  https://www.zillow.com/homedetails/1735-W-Dive...  
    119  https://www.zillow.com/homedetails/1881-N-Poe-...  
    120  https://www.zillow.com/homedetails/1212-N-Lake...  
    121  https://www.zillow.com/homedetails/714-W-Everg...  
    122  https://www.zillow.com/homedetails/849-N-Frank...  
    
    [123 rows x 8 columns]
    

# Investment Cash Flow Calculation Functions

The following two code blocks represent essential functions in the calculation of investment cash flows. These functions are specifically designed to retrieve the average and minimum comparable rent prices for rental listings closest to a specified property for sale, as determined by user input.

To accomplish this, the functions take into account latitude, longitude, and the number of bedrooms to filter rental listings that match the specified bedroom count and then sort them based on proximity. The user can define the comp_amount parameter, which determines how many of the closest rental listings they want to consider in their cash flow calculations. This flexibility is crucial because different areas may have varying densities of rental properties, allowing for accurate and location-specific cash flow assessments.


```python
#BEST MODEL
def AvgCompPrice( latitude, longitude, bedrooms, comp_amount): 
        df = ZillowRentScrape
        #deletes all non matching beds with only props that have equal beds
        df = df.loc[df["bedrooms"] == bedrooms ]
        #assigns column that messures difference between lat & long and adds them
        df['DistanceMessure'] = np.sqrt((df['latitude'] - latitude)**2 + (df['longitude'] - longitude)**2)
        #sorts values by Distance messures
        df = df.sort_values(by=['DistanceMessure'], ascending=True)
        #grabs given amount of values which are closest comps
        Comps = df['price'].head(comp_amount)
        #averages price of comps
        CompAvg = Comps.mean()
     
        #print (df)
        #print(Comps)
        #print(CompAvg)
        return (CompAvg)
```


```python
#BEST MODEL
def MinCompPrice(latitude, longitude, bedrooms, comp_amount):
        df = ZillowRentScrape
        #deletes all non matching beds with only props that have equal beds
        df = df.loc[df["bedrooms"] == bedrooms ]
        #assigns column that messures difference between lat & long and adds them
        df['DistanceMessure'] = np.sqrt((df['latitude'] - latitude)**2 + (df['longitude'] - longitude)**2)
        #sorts values by Distance messures
        df = df.sort_values(by=['DistanceMessure'], ascending=True)
        #grabs given amount of values which are closest comps
        Comps = df['price'].head(comp_amount)
        #min price of comps
        CompLow = Comps.min()
     
        #print (df)
        #print(Comps)
        #print(CompLow)
        return (CompLow)
```

# Website-Based Rental Coordinates Selection

The code block below introduces an additional feature that offers users the flexibility to choose the source website for obtaining longitude and latitude coordinates for rental listings. These coordinates can slightly vary between websites like Zillow and Redfin, and neither platform has demonstrated consistent superiority in accuracy. This capability empowers users to select the most suitable website for their specific needs, ensuring precision in geographic data retrieval.


```python
#MUST BE RAN
#MERGE FILE FOR USING ZILLOW LAT/LONG COMP CALC
Merge1 = RedfinBuyScrape[["address", "estimate_mortgage"]].copy()
print(Merge1)
print (ZillowBuyScrape)
Merged = pd.merge(ZillowBuyScrape,Merge1 ,on='address',how='inner')
Merged.rename(columns={'area': 'area_sqft'}, inplace=True)
Merged['area_sqft'] = Merged['area_sqft'].str.replace(' sqft', '').astype(float)


print(Merged)
```

                                              address  estimate_mortgage
    0     208 W Washington St #709, Chicago, IL 60606         3220.87720
    1       500 W Superior St #801, Chicago, IL 60654         6188.72800
    2        400 W Ontario St #505, Chicago, IL 60654         3906.89612
    3      333 N Jefferson St #303, Chicago, IL 60661         4715.08520
    4      500 W Superior St #1411, Chicago, IL 60654         5744.34000
    ..                                            ...                ...
    67       834 N Wood St Unit 3S, Chicago, IL 60622         3788.75412
    68  714 W Evergreen Ave Unit A, Chicago, IL 60610         4452.93000
    69     21 W Goethe St Unit 8DE, Chicago, IL 60610         4542.59012
    70  1009 N Oakley Blvd Unit 1W, Chicago, IL 60622         2652.86400
    71      401 E Ontario St #1609, Chicago, IL 60611         4926.99000
    
    [72 rows x 2 columns]
                                              address   latitude  longitude  \
    0        200 W Grand Ave #1804, Chicago, IL 60654  41.892040 -87.634476   
    1      700 N Larrabee St #1715, Chicago, IL 60654  41.894608 -87.643090   
    2    208 W Washington St #1510, Chicago, IL 60606  41.883335 -87.634190   
    3     330 S Michigan Ave #1809, Chicago, IL 60604  41.877884 -87.624410   
    4      333 N Jefferson St #303, Chicago, IL 60661  41.887800 -87.642410   
    ..                                            ...        ...        ...   
    118  1735 W Diversey Pkwy #311, Chicago, IL 60614  41.931890 -87.672690   
    119           1881 N Poe St #I, Chicago, IL 60614  41.916195 -87.654140   
    120  1212 N Lake Shore Dr #8CN, Chicago, IL 60610  41.904552 -87.625496   
    121     714 W Evergreen Ave #A, Chicago, IL 60610  41.907238 -87.647660   
    122     849 N Franklin St #404, Chicago, IL 60610  41.897804 -87.635430   
    
            price  bathrooms  bedrooms         area  \
    0    435000.0        2.0       2.0  1127.0 sqft   
    1    495000.0        2.0       2.0  1250.0 sqft   
    2    335000.0        2.0       2.0  1300.0 sqft   
    3    650000.0        3.0       3.0     0.0 sqft   
    4    479000.0        2.0       2.0  1400.0 sqft   
    ..        ...        ...       ...          ...   
    118  399900.0        2.0       2.0  1150.0 sqft   
    119  635000.0        2.0       3.0  1800.0 sqft   
    120  649000.0        2.0       2.0  1850.0 sqft   
    121  475000.0        2.0       2.0  1260.0 sqft   
    122  549900.0        2.0       2.0  1475.0 sqft   
    
                                              property_url  
    0    https://www.zillow.com/homedetails/200-W-Grand...  
    1    https://www.zillow.com/homedetails/700-N-Larra...  
    2    https://www.zillow.com/homedetails/208-W-Washi...  
    3    https://www.zillow.com/homedetails/330-S-Michi...  
    4    https://www.zillow.com/homedetails/333-N-Jeffe...  
    ..                                                 ...  
    118  https://www.zillow.com/homedetails/1735-W-Dive...  
    119  https://www.zillow.com/homedetails/1881-N-Poe-...  
    120  https://www.zillow.com/homedetails/1212-N-Lake...  
    121  https://www.zillow.com/homedetails/714-W-Everg...  
    122  https://www.zillow.com/homedetails/849-N-Frank...  
    
    [123 rows x 8 columns]
                                               address   latitude  longitude  \
    0         200 W Grand Ave #1804, Chicago, IL 60654  41.892040 -87.634476   
    1       700 N Larrabee St #1715, Chicago, IL 60654  41.894608 -87.643090   
    2     208 W Washington St #1510, Chicago, IL 60606  41.883335 -87.634190   
    3      330 S Michigan Ave #1809, Chicago, IL 60604  41.877884 -87.624410   
    4       333 N Jefferson St #303, Chicago, IL 60661  41.887800 -87.642410   
    5      212 W Washington St #802, Chicago, IL 60606  41.883347 -87.634030   
    6       550 N Kingsbury St #110, Chicago, IL 60654  41.891860 -87.640590   
    7       550 N Kingsbury St #420, Chicago, IL 60654  41.891860 -87.640590   
    8            60 W Erie St #1201, Chicago, IL 60654  41.894295 -87.630455   
    9        757 N Orleans St #1312, Chicago, IL 60654  41.896164 -87.636700   
    10         660 W Wayman St #307, Chicago, IL 60661  41.887638 -87.645090   
    11     600 N Kingsbury St #1607, Chicago, IL 60654  41.892920 -87.641594   
    12        411 W Ontario St #612, Chicago, IL 60654  41.892975 -87.639660   
    13        333 W Hubbard St #605, Chicago, IL 60654  41.889545 -87.636130   
    14    208 W Washington St #1213, Chicago, IL 60606  41.883335 -87.634190   
    15           33 W Huron St #509, Chicago, IL 60654  41.894615 -87.629100   
    16  737 W Washington Blvd #2908, Chicago, IL 60661  41.882866 -87.646614   
    17          303 W Ohio St #1803, Chicago, IL 60654  41.892120 -87.636040   
    18         657 W Fulton St #705, Chicago, IL 60661  41.886500 -87.645120   
    19     208 W Washington St #709, Chicago, IL 60606  41.883335 -87.634190   
    20      500 W Superior St #1411, Chicago, IL 60654  41.895775 -87.642220   
    21       700 N Larrabee St #115, Chicago, IL 60654  41.894608 -87.643090   
    22       630 N Franklin St #905, Chicago, IL 60654  41.893406 -87.635864   
    23       500 W Superior St #801, Chicago, IL 60654  41.895775 -87.642220   
    24         60 E Monroe St #3803, Chicago, IL 60603  41.881485 -87.625670   
    25        400 W Ontario St #607, Chicago, IL 60654  41.893356 -87.638960   
    26        400 W Ontario St #707, Chicago, IL 60654  41.893356 -87.638960   
    27       230 W Division St #901, Chicago, IL 60610  41.904030 -87.635840   
    28        155 N Harbor Dr #2302, Chicago, IL 60601  41.884920 -87.614790   
    29            1102 N Wood St #3, Chicago, IL 60622  41.901653 -87.672670   
    30               2701 W Iowa St, Chicago, IL 60622  41.897305 -87.694440   
    31       1512 N Campbell Ave #1, Chicago, IL 60622  41.908913 -87.690000   
    32    1255 N Sandburg Ter #2302, Chicago, IL 60610  41.905310 -87.631720   
    33       222 N Columbus Dr #309, Chicago, IL 60601  41.886520 -87.621020   
    34    1360 N Sandburg Ter #2801, Chicago, IL 60610  41.907047 -87.632220   
    
           price  bathrooms  bedrooms  area_sqft  \
    0   435000.0        2.0       2.0     1127.0   
    1   495000.0        2.0       2.0     1250.0   
    2   335000.0        2.0       2.0     1300.0   
    3   650000.0        3.0       3.0        0.0   
    4   479000.0        2.0       2.0     1400.0   
    5   299500.0        2.0       2.0        0.0   
    6   569000.0        3.0       2.0     1500.0   
    7   575000.0        2.0       2.0     1460.0   
    8   514900.0        2.0       2.0     1450.0   
    9   529000.0        2.0       2.0     1248.0   
    10  399000.0        2.0       2.0        0.0   
    11  465000.0        2.0       2.0     1220.0   
    12  439900.0        2.0       2.0     1282.0   
    13  599000.0        2.0       2.0     1200.0   
    14  369900.0        2.0       2.0     1140.0   
    15  600000.0        2.0       2.0     1400.0   
    16  625000.0        2.0       2.0     1781.0   
    17  540000.0        2.0       2.0     1145.0   
    18  500000.0        2.0       2.0     1200.0   
    19  319000.0        1.0       2.0      850.0   
    20  550000.0        2.0       2.0     1265.0   
    21  459000.0        2.0       2.0        0.0   
    22  474900.0        2.0       2.0     1240.0   
    23  560000.0        3.0       2.0        0.0   
    24  700000.0        2.0       2.0     1600.0   
    25  350000.0        2.0       2.0     1100.0   
    26  350000.0        2.0       2.0        0.0   
    27  499000.0        2.0       2.0        0.0   
    28  415000.0        2.0       2.0     1248.0   
    29  499000.0        2.0       2.0        0.0   
    30  399000.0        2.0       4.0        NaN   
    31  499900.0        3.0       3.0     2320.0   
    32  469000.0        2.0       2.0     1078.0   
    33  398000.0        2.0       2.0     1225.0   
    34  350000.0        2.0       2.0     1100.0   
    
                                             property_url  estimate_mortgage  
    0   https://www.zillow.com/homedetails/200-W-Grand...         4570.37800  
    1   https://www.zillow.com/homedetails/700-N-Larra...         4999.70600  
    2   https://www.zillow.com/homedetails/208-W-Washi...         3938.49800  
    3   https://www.zillow.com/homedetails/330-S-Michi...         7367.22000  
    4   https://www.zillow.com/homedetails/333-N-Jeffe...         4715.08520  
    5   https://www.zillow.com/homedetails/212-W-Washi...         3379.87060  
    6   https://www.zillow.com/homedetails/550-N-Kings...         5463.57720  
    7   https://www.zillow.com/homedetails/550-N-Kings...         5826.81000  
    8   https://www.zillow.com/homedetails/60-W-Erie-S...         5665.32812  
    9   https://www.zillow.com/homedetails/757-N-Orlea...         5271.02520  
    10  https://www.zillow.com/homedetails/660-W-Wayma...         3640.98120  
    11  https://www.zillow.com/homedetails/600-N-Kings...         4655.54200  
    12  https://www.zillow.com/homedetails/411-W-Ontar...         4621.91812  
    13  https://www.zillow.com/homedetails/333-W-Hubba...         5887.74120  
    14  https://www.zillow.com/homedetails/208-W-Washi...         4051.20212  
    15  https://www.zillow.com/homedetails/33-W-Huron-...         5823.28000  
    16  https://www.zillow.com/homedetails/737-W-Washi...         7136.75000  
    17  https://www.zillow.com/homedetails/303-W-Ohio-...         5514.95200  
    18  https://www.zillow.com/homedetails/657-W-Fulto...         4694.40000  
    19  https://www.zillow.com/homedetails/208-W-Washi...         3220.87720  
    20  https://www.zillow.com/homedetails/500-W-Super...         5744.34000  
    21  https://www.zillow.com/homedetails/700-N-Larra...         4541.30920  
    22  https://www.zillow.com/homedetails/630-N-Frank...         4614.77612  
    23  https://www.zillow.com/homedetails/500-W-Super...         6188.72800  
    24  https://www.zillow.com/homedetails/60-E-Monroe...         7491.16000  
    25  https://www.zillow.com/homedetails/400-W-Ontar...         3554.58000  
    26  https://www.zillow.com/homedetails/400-W-Ontar...         3658.58000  
    27  https://www.zillow.com/homedetails/230-W-Divis...         5401.86120  
    28  https://www.zillow.com/homedetails/155-N-Harbo...         4618.60200  
    29  https://www.zillow.com/homedetails/1102-N-Wood...         4557.86120  
    30  https://www.zillow.com/homedetails/2701-W-Iowa...         3575.98120  
    31  https://www.zillow.com/homedetails/1512-N-Camp...         4720.24612  
    32  https://www.zillow.com/homedetails/1255-N-Sand...         4413.69720  
    33  https://www.zillow.com/homedetails/222-N-Colum...         4527.44240  
    34  https://www.zillow.com/homedetails/1360-N-Sand...         4234.58000  
    

# User-Interactive Cash Flow Projection and Visualization

This block facilitates user interaction by collecting essential inputs, including the number of rental comparables to consider in cash flow projections, the preferred coordinates source, and the number of properties to be graphed. Subsequently, it generates informative horizontal bar graphs that offer insights into the highest cash-flowing properties for both 5% and 20% downpayment scenarios. These visualizations empower users to make informed investment decisions by highlighting properties with the most promising cash flow potential.


```python
#Collects user input to decide if comp cordinates are aquired through redfin or zillow
comp_choice = str (input("Do you wish your comps to be found through Zillow or Redfin cordinates? (Type  Zillow or Redfin)  "))
if comp_choice.lower() == 'zillow':
    Final = Merged
elif comp_choice.lower() == 'redfin':
    Final = RedfinBuyScrape
else:
    Final = RedfinBuyScrape
    
#percent_down = str (input("What percent down? (5% or 20%)"))
#if percent_down == '20' or percent_down == '20%':
#    adjustment = 1.0
#elif percent_down == '5' or percent_down == '5%':
#    adjustment = 0.0
#else:
#    adjustment = 0.0
graph_amount = int (input("How many properties do you wish to graph? (1-20)"))
    
comp_amount = int (input("How many comp properties do you wish to consider in cash flow calculations?   "))


Final['Rent est Low'] =Final.apply(lambda row: float (MinCompPrice(row['latitude'], row['longitude'], row['bedrooms'], comp_amount)), axis =1)
Final['Rent est Avg'] =Final.apply(lambda row: float (AvgCompPrice(row['latitude'], row['longitude'], row['bedrooms'], comp_amount)), axis =1)

#Calls min or avg comp price funtion and subtracts estimate mortgage for cash flow gauge
Final['Cash Flow Low 5%'] = Final.apply(lambda row: float (MinCompPrice(row['latitude'], row['longitude'], row['bedrooms'], comp_amount))- row['estimate_mortgage'], axis=1)
Final['Cash Flow Avg 5%'] = Final.apply(lambda row: float (AvgCompPrice(row['latitude'], row['longitude'], row['bedrooms'], comp_amount))- row['estimate_mortgage'], axis=1)
Final['Cash Flow Low 20%'] = Final.apply(lambda row: float (MinCompPrice(row['latitude'], row['longitude'], row['bedrooms'], comp_amount))- (row['estimate_mortgage']-(row['price'] * 0.0015388)), axis=1)
Final['Cash Flow Avg 20%'] = Final.apply(lambda row: float (AvgCompPrice(row['latitude'], row['longitude'], row['bedrooms'], comp_amount))- (row['estimate_mortgage']-(row['price'] * 0.0015388)), axis=1)

#Final.rename(columns={'estimate_mortgage': 'estimate mortgage @5%'}, inplace=True)
Final['estimate mortgage @20%'] = Final['estimate_mortgage'] - (Final['price'] * 0.0015388)

#sorts by highest cash flowing properties
Final = Final.sort_values(by=['Cash Flow Low 5%'], ascending=False)




#Graphing highest cash flow propeties
#Just view plot for explanation
visual1 = Final
visual1 = visual1.set_index('address')
visual1.sort_values('Cash Flow Low 5%', ascending=True, inplace=True)
visual1['colors'] = ['red' if x < 0 else 'green' for x in visual1['Cash Flow Low 5%']]
visual1 = visual1.iloc[-graph_amount:]



plt.figure(figsize=(14,10), dpi= 80)
plt.hlines(y=visual1.index, xmin=0, xmax = visual1['Cash Flow Low 5%'], color=visual1.colors, alpha=0.4, linewidth=5)
plt.title('Monthly Cash Flow Low Analysis 5% Down', fontdict={'size':20})
plt.yticks(visual1.index, fontsize=12)
plt.grid(linestyle='--', alpha=0.5)
plt.gca().set(ylabel='Properties', xlabel= 'Monthly Cash Flow in USD')

plt.show

visual2 = Final
visual2 = visual2.set_index('address')
visual2.sort_values('Cash Flow Low 20%', ascending=True, inplace=True)
visual2['colors'] = ['red' if x < 0 else 'green' for x in visual2['Cash Flow Low 20%']]
visual2 = visual2.iloc[-graph_amount:]



plt.figure(figsize=(14,10), dpi= 80)
plt.hlines(y=visual2.index, xmin=0, xmax = visual2['Cash Flow Low 20%'], color=visual2.colors, alpha=0.4, linewidth=5)
plt.title('Monthly Cash Flow Low Analysis 20% Down', fontdict={'size':20})
plt.yticks(visual2.index, fontsize=12)
plt.grid(linestyle='--', alpha=0.5)
plt.gca().set(ylabel='Properties', xlabel= 'Monthly Cash Flow in USD')

plt.show


```

    Do you wish your comps to be found through Zillow or Redfin cordinates? (Type  Zillow or Redfin)  Zillow
    How many properties do you wish to graph? (1-20)20
    How many comp properties do you wish to consider in cash flow calculations?   8
    




    <function matplotlib.pyplot.show(close=None, block=None)>




    
![png](output_12_2.png)
    



    
![png](output_12_3.png)
    


# Geopandas Support Functions

The following functions are designed to support the Geopandas section of the code. The first function takes a string of X,Y coordinates and transforms them into points that can be easily interpreted by Geopandas. This conversion is crucial for geospatial data analysis.

The second function, akin to the rental minimum and average functions mentioned earlier, serves a different purpose by providing longitude and latitude coordinates of the closest properties. This function enables users to identify and work with geospatial data related to nearby properties, which is valuable for various geographical analyses.


```python
#breaks strings to be read by geo pandas
def string_to_point(string):
    x, y = map(float, string.split())
    return Point(x, y)
```


```python
def CompLocater(latitude, longitude, bedrooms, comp_amount):
        df = ZillowRentScrape
        #deletes all non matching beds with only props that have equal beds
        df = df.loc[df["bedrooms"] == bedrooms ]
        #assigns column that messures difference between lat & long and adds them
        df['DistanceMessure'] = np.sqrt((df['latitude'] - latitude)**2 + (df['longitude'] - longitude)**2)
        #sorts values by Distance messures
        df = df.sort_values(by=['DistanceMessure'], ascending=True)
        #grabs top 20 values
        df = df.head(20)
        #sets dataframe equal to lat long cordinates of the index value passed by loop
        df = df[['latitude']].iloc[comp_amount],df[['longitude']].iloc[comp_amount]
        
        
 
        
        
        #print(df) 
        #print (Cord1)
        #print(Comps)
        #print(CompLow)
        return (df)
```

# Comparative Rental Location Visualization

The code block below introduces a feature that empowers users to visualize the geographical locations of comparable rental listings in relation to the highest-earning properties they've selected. This visualization offers valuable insights into the distribution of rental properties, enabling users to identify potential outliers. In these graphical representations, the property for sale is distinctly marked with a red star, while the rental properties are represented as blue dots. This visualization aids in better understanding the spatial relationship between rental listings and the potential properties.



```python
#Maps the highest cash flowing properties and maps the comps
    #this gives a idea of comp concentration
plot_amount = int (input("Of the highest cash flowing properties, how many do you wish to have mapped with comp locations?   "))

#Dataframe for possible propeties to buy
Mapping2 = Final.copy()
Mapping2['Coordinates'] = list(zip(Mapping2.longitude, Mapping2.latitude))
Mapping2['Coordinates'] = Mapping2['Coordinates'].apply(Point)



#Dataframe for comp properties, calls CompLocater to get latlong
comp_amount_input = comp_amount
best = Mapping2.head(plot_amount)
for i in range(comp_amount_input):
    best[i] = best.apply(lambda row: CompLocater(row['latitude'], row['longitude'], row['bedrooms'], i), axis=1)
    {i+1}

        

#extracts long/lat from dataframe above
for i in range(comp_amount_input):
    best[i] = best[i].astype(str)
    best[i] = (best[i].str.extract(r'longitude\s+([\d.-]+)')) + '  ' + (best[i].str.extract(r'latitude\s+([\d.-]+)'))
    {i+1}

for i in range(plot_amount):
    CompMapping = geopandas.read_file('geo_export_d916f11b-fbee-4e54-94ee-60bf7ad450ef.shp')
    CompMapping2 = best
    CompMapping2 = CompMapping2.reset_index(drop=True)
    CompMapping2 = CompMapping2.drop(CompMapping2.index.difference([i]))
    graph_scale_1 = CompMapping2['latitude'].iloc[0]
    graph_scale_2 = CompMapping2['longitude'].iloc[0]



    fig, gax = plt.subplots(figsize=(10,10))

    CompMapping['geometry'].plot(ax = gax, edgecolor='black',color='white')
  

    gdf = geopandas.GeoDataFrame(CompMapping2, geometry='Coordinates')
    gax.set_ylim([(graph_scale_1 - 0.01), (graph_scale_1 + 0.01)])
    gax.set_xlim([(graph_scale_2 - 0.01), (graph_scale_2 + 0.01)])
    gdf.plot(ax=gax, color='red', alpha = 0.9,  marker='*', markersize=100)
# i
        
    new_df = CompMapping2.loc[: ,'Coordinates':].copy()
    new_df = new_df.drop('Coordinates', axis=1)
    new_df = new_df.transpose()
    new_df[i] = new_df[i].apply(string_to_point)
    new_df = geopandas.GeoDataFrame(new_df, geometry=i)
    new_df.plot(ax=gax, color='blue', alpha = 0.9)
    plt.title((CompMapping2['address'].iloc[0]), fontdict={'size':20})
    plt.gca().set(ylabel='Latitude (each tick = .15 mile)', xlabel= 'Longitude (each tick = .15 mile)')
    plt.grid(linestyle='--', alpha=0.5) 

    


#
    

    plt.show()
    




    {i+1}
    

    
  










```

    Of the highest cash flowing properties, how many do you wish to have mapped with comp locations?   8
    

    C:\Users\Administrator\AppData\Local\Temp\ipykernel_15384\2940697431.py:16: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      best[i] = best.apply(lambda row: CompLocater(row['latitude'], row['longitude'], row['bedrooms'], i), axis=1)
    C:\Users\Administrator\AppData\Local\Temp\ipykernel_15384\2940697431.py:16: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      best[i] = best.apply(lambda row: CompLocater(row['latitude'], row['longitude'], row['bedrooms'], i), axis=1)
    C:\Users\Administrator\AppData\Local\Temp\ipykernel_15384\2940697431.py:16: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      best[i] = best.apply(lambda row: CompLocater(row['latitude'], row['longitude'], row['bedrooms'], i), axis=1)
    C:\Users\Administrator\AppData\Local\Temp\ipykernel_15384\2940697431.py:16: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      best[i] = best.apply(lambda row: CompLocater(row['latitude'], row['longitude'], row['bedrooms'], i), axis=1)
    C:\Users\Administrator\AppData\Local\Temp\ipykernel_15384\2940697431.py:16: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      best[i] = best.apply(lambda row: CompLocater(row['latitude'], row['longitude'], row['bedrooms'], i), axis=1)
    C:\Users\Administrator\AppData\Local\Temp\ipykernel_15384\2940697431.py:16: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      best[i] = best.apply(lambda row: CompLocater(row['latitude'], row['longitude'], row['bedrooms'], i), axis=1)
    C:\Users\Administrator\AppData\Local\Temp\ipykernel_15384\2940697431.py:16: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      best[i] = best.apply(lambda row: CompLocater(row['latitude'], row['longitude'], row['bedrooms'], i), axis=1)
    C:\Users\Administrator\AppData\Local\Temp\ipykernel_15384\2940697431.py:16: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      best[i] = best.apply(lambda row: CompLocater(row['latitude'], row['longitude'], row['bedrooms'], i), axis=1)
    C:\Users\Administrator\AppData\Local\Temp\ipykernel_15384\2940697431.py:23: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      best[i] = best[i].astype(str)
    C:\Users\Administrator\AppData\Local\Temp\ipykernel_15384\2940697431.py:24: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      best[i] = (best[i].str.extract(r'longitude\s+([\d.-]+)')) + '  ' + (best[i].str.extract(r'latitude\s+([\d.-]+)'))
    C:\Users\Administrator\AppData\Local\Temp\ipykernel_15384\2940697431.py:23: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      best[i] = best[i].astype(str)
    C:\Users\Administrator\AppData\Local\Temp\ipykernel_15384\2940697431.py:24: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      best[i] = (best[i].str.extract(r'longitude\s+([\d.-]+)')) + '  ' + (best[i].str.extract(r'latitude\s+([\d.-]+)'))
    C:\Users\Administrator\AppData\Local\Temp\ipykernel_15384\2940697431.py:23: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      best[i] = best[i].astype(str)
    C:\Users\Administrator\AppData\Local\Temp\ipykernel_15384\2940697431.py:24: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      best[i] = (best[i].str.extract(r'longitude\s+([\d.-]+)')) + '  ' + (best[i].str.extract(r'latitude\s+([\d.-]+)'))
    C:\Users\Administrator\AppData\Local\Temp\ipykernel_15384\2940697431.py:23: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      best[i] = best[i].astype(str)
    C:\Users\Administrator\AppData\Local\Temp\ipykernel_15384\2940697431.py:24: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      best[i] = (best[i].str.extract(r'longitude\s+([\d.-]+)')) + '  ' + (best[i].str.extract(r'latitude\s+([\d.-]+)'))
    C:\Users\Administrator\AppData\Local\Temp\ipykernel_15384\2940697431.py:23: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      best[i] = best[i].astype(str)
    C:\Users\Administrator\AppData\Local\Temp\ipykernel_15384\2940697431.py:24: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      best[i] = (best[i].str.extract(r'longitude\s+([\d.-]+)')) + '  ' + (best[i].str.extract(r'latitude\s+([\d.-]+)'))
    C:\Users\Administrator\AppData\Local\Temp\ipykernel_15384\2940697431.py:23: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      best[i] = best[i].astype(str)
    C:\Users\Administrator\AppData\Local\Temp\ipykernel_15384\2940697431.py:24: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      best[i] = (best[i].str.extract(r'longitude\s+([\d.-]+)')) + '  ' + (best[i].str.extract(r'latitude\s+([\d.-]+)'))
    C:\Users\Administrator\AppData\Local\Temp\ipykernel_15384\2940697431.py:23: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      best[i] = best[i].astype(str)
    C:\Users\Administrator\AppData\Local\Temp\ipykernel_15384\2940697431.py:24: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      best[i] = (best[i].str.extract(r'longitude\s+([\d.-]+)')) + '  ' + (best[i].str.extract(r'latitude\s+([\d.-]+)'))
    C:\Users\Administrator\AppData\Local\Temp\ipykernel_15384\2940697431.py:23: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      best[i] = best[i].astype(str)
    C:\Users\Administrator\AppData\Local\Temp\ipykernel_15384\2940697431.py:24: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      best[i] = (best[i].str.extract(r'longitude\s+([\d.-]+)')) + '  ' + (best[i].str.extract(r'latitude\s+([\d.-]+)'))
    


    
![png](output_17_2.png)
    



    
![png](output_17_3.png)
    



    
![png](output_17_4.png)
    



    
![png](output_17_5.png)
    



    
![png](output_17_6.png)
    



    
![png](output_17_7.png)
    



    
![png](output_17_8.png)
    



    
![png](output_17_9.png)
    


# Machine Learning Rent Prediction Models

In this section is a machine learning framework that empowers users to select a specific property and employ diverse learning models to predict its rent. The primary advantage of these machine learning models lies in their ability to incorporate property square footages, a factor not accounted for in previous rent projections. This original omission stems from the nuanced relationship between rent and square footage, where a fixed multiplier could introduce inaccuracies due to varying correlation strengths.

It is important to note that these implementations have room for significant improvement through correct scaling and the potential narrowing down to one or two optimal models. The option to choose from various learning models and access Mean Squared Error (MSE) and R^2 metrics is intended for users who possess an understanding of the advantages and limitations of each model, enabling them to make informed decisions based on their specific needs.


```python


#Further investment analysis
Further_Analysis = Final.copy()
Further_Analysis = Further_Analysis.reset_index(drop=True)
print(Further_Analysis)
User_Choice = int (input("Which Property do you wish to futher analyze? Type the property identifier found to the left of the address.\nFor model to properly work, choose only properties with a value in area_sqft\n"))
User_Choice2 = int (input("For rent prediction,\n Type 1 for Random Forest Regressor\n Type 2 for Support Vector Regression\n Type 3 for Gradient Boosting Regressor\n Type 4 for Bayesian Ridge\n"))

#User_Choice_Df = Further_Analysis.iloc[User_Choice].copy()
User_Choice_Df = Further_Analysis.copy()
User_Choice_Df = User_Choice_Df.iloc[[User_Choice]]

#RandomForestRegressorModel
#New DF for userchoice tailored to model input
User_Choice_RF = User_Choice_Df[['longitude', 'latitude', 'bedrooms', 'bathrooms', 'area_sqft']].copy()
User_Choice_RF.rename(columns={'area_sqft': 'area'}, inplace=True)

#New DF for train/test data
Predict1 = ZillowRentScrape[['longitude','latitude','price','area','bedrooms','bathrooms']].copy()

#Drops na for no model errors
Predict1.dropna(inplace=True)

Predict1['area'] = Predict1['area'].str.replace(' sqft', '').astype(float)

#important features
features = ['latitude', 'longitude', 'bathrooms', 'bedrooms', 'area']
target = 'price'

#setting train/test data
X_train, X_test, y_train, y_test = train_test_split(Predict1[features], Predict1[target], test_size=0.16, random_state=42)



# Create and train the Random Forest Regression model
### n_estimators amount of probability trees
if User_Choice2 == 1 :
    model = RandomForestRegressor(n_estimators=100, random_state=42)
if User_Choice2 == 2 :
    model = SVR()
if User_Choice2 == 3 :
    model =  GradientBoostingRegressor(random_state=42)
if User_Choice2 == 4 :
    model =  BayesianRidge()
    
model.fit(X_train, y_train)


# Make predictions on the test set
y_pred = model.predict(X_test)



# Calculate mean squared error as a performance metric
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
print('Mean Squared Error:', mse)
print('R-squared:', r2)


input_df = pd.DataFrame(User_Choice_RF, columns=['latitude', 'longitude', 'bathrooms', 'bedrooms', 'area'])




predicted_price = model.predict(input_df)



if User_Choice2 == 1 :
    print('Random Forest Regressor Predicted Rent:', predicted_price)
if User_Choice2 == 2 :
    print('Support Vector Regression Predicted Rent:', predicted_price)
if User_Choice2 == 3 :
    print('Gradient Boosting Regressor Predicted Rent:', predicted_price)
if User_Choice2 == 4 :
    print('Bayesian Ridge Predicted Rent:', predicted_price)




At5 = predicted_price - User_Choice_Df['estimate_mortgage'].iloc[0]
At20 =  predicted_price - ((User_Choice_Df['estimate_mortgage'].iloc[0]) - (User_Choice_Df['price'].iloc[0] * 0.0015388))

print('Selected ML model predicted cash flow at 5% down:', At5)
print('Selected ML model cash flow at 20% down:', At20)


print("-------------------------------------------------------------------------------------------")






```

                                               address   latitude  longitude  \
    0         400 W Ontario St #607, Chicago, IL 60654  41.893356 -87.638960   
    1         400 W Ontario St #707, Chicago, IL 60654  41.893356 -87.638960   
    2      208 W Washington St #709, Chicago, IL 60606  41.883335 -87.634190   
    3      212 W Washington St #802, Chicago, IL 60606  41.883347 -87.634030   
    4         200 W Grand Ave #1804, Chicago, IL 60654  41.892040 -87.634476   
    5          660 W Wayman St #307, Chicago, IL 60661  41.887638 -87.645090   
    6                2701 W Iowa St, Chicago, IL 60622  41.897305 -87.694440   
    7        630 N Franklin St #905, Chicago, IL 60654  41.893406 -87.635864   
    8         411 W Ontario St #612, Chicago, IL 60654  41.892975 -87.639660   
    9      600 N Kingsbury St #1607, Chicago, IL 60654  41.892920 -87.641594   
    10        155 N Harbor Dr #2302, Chicago, IL 60601  41.884920 -87.614790   
    11    208 W Washington St #1510, Chicago, IL 60606  41.883335 -87.634190   
    12       222 N Columbus Dr #309, Chicago, IL 60601  41.886520 -87.621020   
    13    208 W Washington St #1213, Chicago, IL 60606  41.883335 -87.634190   
    14    1255 N Sandburg Ter #2302, Chicago, IL 60610  41.905310 -87.631720   
    15    1360 N Sandburg Ter #2801, Chicago, IL 60610  41.907047 -87.632220   
    16       757 N Orleans St #1312, Chicago, IL 60654  41.896164 -87.636700   
    17       700 N Larrabee St #115, Chicago, IL 60654  41.894608 -87.643090   
    18         657 W Fulton St #705, Chicago, IL 60661  41.886500 -87.645120   
    19          303 W Ohio St #1803, Chicago, IL 60654  41.892120 -87.636040   
    20      333 N Jefferson St #303, Chicago, IL 60661  41.887800 -87.642410   
    21      550 N Kingsbury St #110, Chicago, IL 60654  41.891860 -87.640590   
    22      700 N Larrabee St #1715, Chicago, IL 60654  41.894608 -87.643090   
    23        333 W Hubbard St #605, Chicago, IL 60654  41.889545 -87.636130   
    24      550 N Kingsbury St #420, Chicago, IL 60654  41.891860 -87.640590   
    25           60 W Erie St #1201, Chicago, IL 60654  41.894295 -87.630455   
    26            1102 N Wood St #3, Chicago, IL 60622  41.901653 -87.672670   
    27           33 W Huron St #509, Chicago, IL 60654  41.894615 -87.629100   
    28       1512 N Campbell Ave #1, Chicago, IL 60622  41.908913 -87.690000   
    29       230 W Division St #901, Chicago, IL 60610  41.904030 -87.635840   
    30      500 W Superior St #1411, Chicago, IL 60654  41.895775 -87.642220   
    31       500 W Superior St #801, Chicago, IL 60654  41.895775 -87.642220   
    32  737 W Washington Blvd #2908, Chicago, IL 60661  41.882866 -87.646614   
    33     330 S Michigan Ave #1809, Chicago, IL 60604  41.877884 -87.624410   
    34         60 E Monroe St #3803, Chicago, IL 60603  41.881485 -87.625670   
    
           price  bathrooms  bedrooms  area_sqft  \
    0   350000.0        2.0       2.0     1100.0   
    1   350000.0        2.0       2.0        0.0   
    2   319000.0        1.0       2.0      850.0   
    3   299500.0        2.0       2.0        0.0   
    4   435000.0        2.0       2.0     1127.0   
    5   399000.0        2.0       2.0        0.0   
    6   399000.0        2.0       4.0        NaN   
    7   474900.0        2.0       2.0     1240.0   
    8   439900.0        2.0       2.0     1282.0   
    9   465000.0        2.0       2.0     1220.0   
    10  415000.0        2.0       2.0     1248.0   
    11  335000.0        2.0       2.0     1300.0   
    12  398000.0        2.0       2.0     1225.0   
    13  369900.0        2.0       2.0     1140.0   
    14  469000.0        2.0       2.0     1078.0   
    15  350000.0        2.0       2.0     1100.0   
    16  529000.0        2.0       2.0     1248.0   
    17  459000.0        2.0       2.0        0.0   
    18  500000.0        2.0       2.0     1200.0   
    19  540000.0        2.0       2.0     1145.0   
    20  479000.0        2.0       2.0     1400.0   
    21  569000.0        3.0       2.0     1500.0   
    22  495000.0        2.0       2.0     1250.0   
    23  599000.0        2.0       2.0     1200.0   
    24  575000.0        2.0       2.0     1460.0   
    25  514900.0        2.0       2.0     1450.0   
    26  499000.0        2.0       2.0        0.0   
    27  600000.0        2.0       2.0     1400.0   
    28  499900.0        3.0       3.0     2320.0   
    29  499000.0        2.0       2.0        0.0   
    30  550000.0        2.0       2.0     1265.0   
    31  560000.0        3.0       2.0        0.0   
    32  625000.0        2.0       2.0     1781.0   
    33  650000.0        3.0       3.0        0.0   
    34  700000.0        2.0       2.0     1600.0   
    
                                             property_url  estimate_mortgage  \
    0   https://www.zillow.com/homedetails/400-W-Ontar...         3554.58000   
    1   https://www.zillow.com/homedetails/400-W-Ontar...         3658.58000   
    2   https://www.zillow.com/homedetails/208-W-Washi...         3220.87720   
    3   https://www.zillow.com/homedetails/212-W-Washi...         3379.87060   
    4   https://www.zillow.com/homedetails/200-W-Grand...         4570.37800   
    5   https://www.zillow.com/homedetails/660-W-Wayma...         3640.98120   
    6   https://www.zillow.com/homedetails/2701-W-Iowa...         3575.98120   
    7   https://www.zillow.com/homedetails/630-N-Frank...         4614.77612   
    8   https://www.zillow.com/homedetails/411-W-Ontar...         4621.91812   
    9   https://www.zillow.com/homedetails/600-N-Kings...         4655.54200   
    10  https://www.zillow.com/homedetails/155-N-Harbo...         4618.60200   
    11  https://www.zillow.com/homedetails/208-W-Washi...         3938.49800   
    12  https://www.zillow.com/homedetails/222-N-Colum...         4527.44240   
    13  https://www.zillow.com/homedetails/208-W-Washi...         4051.20212   
    14  https://www.zillow.com/homedetails/1255-N-Sand...         4413.69720   
    15  https://www.zillow.com/homedetails/1360-N-Sand...         4234.58000   
    16  https://www.zillow.com/homedetails/757-N-Orlea...         5271.02520   
    17  https://www.zillow.com/homedetails/700-N-Larra...         4541.30920   
    18  https://www.zillow.com/homedetails/657-W-Fulto...         4694.40000   
    19  https://www.zillow.com/homedetails/303-W-Ohio-...         5514.95200   
    20  https://www.zillow.com/homedetails/333-N-Jeffe...         4715.08520   
    21  https://www.zillow.com/homedetails/550-N-Kings...         5463.57720   
    22  https://www.zillow.com/homedetails/700-N-Larra...         4999.70600   
    23  https://www.zillow.com/homedetails/333-W-Hubba...         5887.74120   
    24  https://www.zillow.com/homedetails/550-N-Kings...         5826.81000   
    25  https://www.zillow.com/homedetails/60-W-Erie-S...         5665.32812   
    26  https://www.zillow.com/homedetails/1102-N-Wood...         4557.86120   
    27  https://www.zillow.com/homedetails/33-W-Huron-...         5823.28000   
    28  https://www.zillow.com/homedetails/1512-N-Camp...         4720.24612   
    29  https://www.zillow.com/homedetails/230-W-Divis...         5401.86120   
    30  https://www.zillow.com/homedetails/500-W-Super...         5744.34000   
    31  https://www.zillow.com/homedetails/500-W-Super...         6188.72800   
    32  https://www.zillow.com/homedetails/737-W-Washi...         7136.75000   
    33  https://www.zillow.com/homedetails/330-S-Michi...         7367.22000   
    34  https://www.zillow.com/homedetails/60-E-Monroe...         7491.16000   
    
        Rent est Low  Rent est Avg  Cash Flow Low 5%  Cash Flow Avg 5%  \
    0         3300.0      3574.875        -254.58000          20.29500   
    1         3300.0      3574.875        -358.58000         -83.70500   
    2         2295.0      3193.000        -925.87720         -27.87720   
    3         2295.0      3193.000       -1084.87060        -186.87060   
    4         3463.0      3799.000       -1107.37800        -771.37800   
    5         2500.0      3307.625       -1140.98120        -333.35620   
    6         2350.0      3224.375       -1225.98120        -351.60620   
    7         3300.0      3615.500       -1314.77612        -999.27612   
    8         3300.0      3587.375       -1321.91812       -1034.54312   
    9         3200.0      3722.000       -1455.54200        -933.54200   
    10        3000.0      3794.250       -1618.60200        -824.35200   
    11        2295.0      3193.000       -1643.49800        -745.49800   
    12        2800.0      3671.625       -1727.44240        -855.81740   
    13        2295.0      3193.000       -1756.20212        -858.20212   
    14        2495.0      3443.375       -1918.69720        -970.32220   
    15        2245.0      3464.875       -1989.58000        -769.70500   
    16        3250.0      3855.000       -2021.02520       -1416.02520   
    17        2500.0      3491.000       -2041.30920       -1050.30920   
    18        2500.0      3557.500       -2194.40000       -1136.90000   
    19        3300.0      3714.250       -2214.95200       -1800.70200   
    20        2500.0      3195.625       -2215.08520       -1519.46020   
    21        3200.0      3559.125       -2263.57720       -1904.45220   
    22        2500.0      3491.000       -2499.70600       -1508.70600   
    23        3310.0      4479.250       -2577.74120       -1408.49120   
    24        3200.0      3559.125       -2626.81000       -2267.68500   
    25        2768.0      3652.375       -2897.32812       -2012.95312   
    26        1600.0      2609.375       -2957.86120       -1948.48620   
    27        2768.0      3652.375       -3055.28000       -2170.90500   
    28        1615.0      2420.000       -3105.24612       -2300.24612   
    29        2200.0      3885.625       -3201.86120       -1516.23620   
    30        2500.0      3585.125       -3244.34000       -2159.21500   
    31        2500.0      3585.125       -3688.72800       -2603.60300   
    32        3409.0      4110.375       -3727.75000       -3026.37500   
    33        2695.0      6768.000       -4672.22000        -599.22000   
    34        2095.0      3536.625       -5396.16000       -3954.53500   
    
        Cash Flow Low 20%  Cash Flow Avg 20%  estimate mortgage @20%  
    0               284.0            558.875                  3016.0  
    1               180.0            454.875                  3120.0  
    2              -435.0            463.000                  2730.0  
    3              -624.0            274.000                  2919.0  
    4              -438.0           -102.000                  3901.0  
    5              -527.0            280.625                  3027.0  
    6              -612.0            262.375                  2962.0  
    7              -584.0           -268.500                  3884.0  
    8              -645.0           -357.625                  3945.0  
    9              -740.0           -218.000                  3940.0  
    10             -980.0           -185.750                  3980.0  
    11            -1128.0           -230.000                  3423.0  
    12            -1115.0           -243.375                  3915.0  
    13            -1187.0           -289.000                  3482.0  
    14            -1197.0           -248.625                  3692.0  
    15            -1451.0           -231.125                  3696.0  
    16            -1207.0           -602.000                  4457.0  
    17            -1335.0           -344.000                  3835.0  
    18            -1425.0           -367.500                  3925.0  
    19            -1384.0           -969.750                  4684.0  
    20            -1478.0           -782.375                  3978.0  
    21            -1388.0          -1028.875                  4588.0  
    22            -1738.0           -747.000                  4238.0  
    23            -1656.0           -486.750                  4966.0  
    24            -1742.0          -1382.875                  4942.0  
    25            -2105.0          -1220.625                  4873.0  
    26            -2190.0          -1180.625                  3790.0  
    27            -2132.0          -1247.625                  4900.0  
    28            -2336.0          -1531.000                  3951.0  
    29            -2434.0           -748.375                  4634.0  
    30            -2398.0          -1312.875                  4898.0  
    31            -2827.0          -1741.875                  5327.0  
    32            -2766.0          -2064.625                  6175.0  
    33            -3672.0            401.000                  6367.0  
    34            -4319.0          -2877.375                  6414.0  
    Which Property do you wish to futher analyze? Type the property identifier found to the left of the address.
    For model to properly work, choose only properties with a value in area_sqft
    0
    For rent prediction,
     Type 1 for Random Forest Regressor
     Type 2 for Support Vector Regression
     Type 3 for Gradient Boosting Regressor
     Type 4 for Bayesian Ridge
    1
    Mean Squared Error: 225340.54444810122
    R-squared: 0.7892830561441742
    Random Forest Regressor Predicted Rent: [3675.68]
    Selected ML model predicted cash flow at 5% down: [121.1]
    Selected ML model cash flow at 20% down: [659.68]
    -------------------------------------------------------------------------------------------
    

# Focused Machine Learning on Comparable Properties


In this unfinished section, the possibility is explored of refining the machine learning model to concentrate solely on learning from the found comparable properties rather than using all the rental properties obtained through scraping. This approach allows for a more targeted and specialized model training process. By narrowing the model's focus to the most relevant and similar properties, it aims to enhance its predictive accuracy, particularly when it comes to predicting rents for specific properties of interest. This potential integration aligns with the idea of tailoring machine learning models for more precise and context-specific predictions


```python
def MLCompDf( latitude, longitude, bedrooms, comp_amount): 
        df = ZillowRentScrape
        #deletes all non matching beds with only props that have equal beds
        df = df.loc[df["bedrooms"] == bedrooms ]
        #assigns column that messures difference between lat & long and adds them
        df['DistanceMessure'] = np.sqrt((df['latitude'] - latitude)**2 + (df['longitude'] - longitude)**2)
        #sorts values by Distance messures
        df = df.sort_values(by=['DistanceMessure'], ascending=True)
        #grabs given amount of values which are closest comps
        Comps = df.head(comp_amount)
        #Comps = df['price'].head(comp_amount)
        #averages price of comps
        #CompAvg = Comps.mean()
     
        #print (df)
        #print(Comps)
        #print(CompAvg)
        return (Comps)
```

# Further Investment Insights and Analytics

This section extends the logic from Real Estate Acquisition Advisor Version 1, offering an in-depth analysis of the selected property within the machine learning models. Users are prompted to provide various inputs, including mortgage duration, holding periods, current interest rates, expenses, rent adjustments, and vacancy rates. Using these user-defined values, the system generates valuable insights into investment scenarios, such as property value projections, Multiple on Invested Capital (MOIC) over time, and neighborhood analytics sourced from Niche. Additionally, it provides trends in property values by zip code through Zillow property data. This comprehensive analysis empowers users with a thorough understanding of their investment's potential and associated financial outcomes.


```python
Int_Rate = float (input("What is the current interest rate?\n"))
D_Payment = float (input ("What percent do you intend to put down? (5 or 20)\n"))
Term = float (input ("What term length? (15 or 30 years)\n"))
ExpenseInc = float (input ("What is the Annual Expense Increase Rate?\n"))
ExpenseInc = ExpenseInc/100.0
RentInc = float (input ("What is the Annual Rent Increase Rate?\n"))
RentInc = RentInc/100.0
Vacancy = float (input ("What is the predicted vacancy?\n"))
Vacancy = Vacancy/100.0
HPI = float (input ("What is the predicted Annual HPI?\n"))
CashOut = int (input ("After how many months do you intend to sell?\nIf you don't, enter 0 \n"))

Term_Months = 0
if Term == 15:
    Term_Months = 181
if Term == 30:
    Term_Months = 361
if CashOut == 0:
    Cashout = Term_Months

    
    

Deep_Analysis = User_Choice_Df.copy()
Deep_Analysis = Deep_Analysis.reset_index(drop=True)
#Deep_Analysis.reset_index(drop=True)
#Deep_Analysis = Deep_Analysis.transpose()
print(Deep_Analysis)


##############################################################################################################
#Niche Rating
AddressWCode = Deep_Analysis.iloc[0, 0]
ZipCode = int(AddressWCode[-5:])
image_path = r'C:\Users\Administrator\Dropbox\Apps\ScrapeHero-Cloud\{}.jpeg'.format(ZipCode)
image = mpimg.imread(image_path)
plt.figure(dpi=250)
plt.imshow(image)
plt.axis('off')
plt.show()

##############################################################################################################
loan_months = list(range(0, Term_Months))


##############################################################################################################
df = pd.DataFrame(index=loan_months)
df['Loan Months'] = loan_months
df['Loan Payment'] = 0.0
df['Loan Interest'] = 0.0
df['Loan Principal'] = 0.0
df['Loan Balance'] = 0.0
df['Cumulative Principal (No Appreciation)'] = 0.0
df['Cumulative Principal (Appreciation)'] = 0.0
df['Expense Index'] = 0.0
df['Rental Index'] = 0.0
df['Rent Value'] = 0.0
df['Net Rent Approximation'] = 0.0
df['Property Level HPI'] = 0.0
df['Home Value'] = 0.0
df['Cash Flow Approximation (Hold)'] = 0.0
df['Cumulative Net Income (Hold)'] = 0.0
df['MOIC (Hold)'] =0.0
df['Cash Flow Approximation (Sell)'] =0.0
df['Cumulative Net Income (Sell)'] =0.0
df['MOIC (Sell)'] =0.0

df.loc[0, 'Loan Balance'] = (1-(D_Payment/100)) * Deep_Analysis['price'].iloc[0]
df.loc[0, 'Loan Payment'] = Deep_Analysis['price'].iloc[0]*(D_Payment/100)
df.loc[0, 'Loan Principal'] = df.loc[0, 'Loan Payment']
df.loc[1:, 'Loan Payment'] =  (npf.pmt((Int_Rate/100)/12, Term_Months-1, -(df.loc[0, 'Loan Balance'])))
df.loc[1:,'Loan Interest'] = npf.ipmt((Int_Rate/100)/12, df['Loan Months'].iloc[1:], Term_Months-1, -(df.loc[0, 'Loan Balance']))
df.loc[1:,'Loan Principal'] = npf.ppmt((Int_Rate/100)/12, df['Loan Months'].iloc[1:], Term_Months-1, -(df.loc[0, 'Loan Balance']))


for i in range(1, len(df)-1):
    df.at[i, 'Loan Balance'] = df.at[i-1, 'Loan Balance'] - (df.at[i, 'Loan Principal'])
    
    
df.loc[0, 'Cumulative Principal (No Appreciation)'] = df.loc[0, 'Loan Principal']

for i in range(1, len(df)):
    df.at[i, 'Cumulative Principal (No Appreciation)'] = df.at[i-1, 'Cumulative Principal (No Appreciation)'] + (df.at[i, 'Loan Principal'])
    
df.loc[0, 'Expense Index'] = 100
for i in range(1, len(df)):
    df.at[i, 'Expense Index'] = df.at[i-1, 'Expense Index'] * (ExpenseInc + 1) ** (1/12)
    
    
df.loc[0, 'Rental Index'] = 100
for i in range(1, len(df)):
    df.at[i, 'Rental Index'] = df.at[i-1, 'Rental Index'] * (RentInc + 1) ** (1/12)

#df.loc[1, 'Rent Value'] = Deep_Analysis.loc[0, 'Rent est Low'] * df.loc[1, 'Rental Index']
for i in range(1, len(df)):
    df.at[i, 'Rent Value'] = Deep_Analysis.loc[0, 'Rent est Low'] * (df.loc[i, 'Rental Index']/100)

df['Net Rent Approximation'] = df['Rent Value'] * (1-Vacancy)

df.loc[0, 'Property Level HPI'] = 1
for i in range(1, len(df)):
    df.at[i, 'Property Level HPI'] = df.at[i-1, 'Property Level HPI'] * ((HPI/100) + 1) ** (1/12)
    
df['Home Value'] =  Deep_Analysis.loc[0, 'price'] * df['Property Level HPI']

df['Cumulative Principal (Appreciation)'] = df['Cumulative Principal (No Appreciation)'] + df['Home Value'] - Deep_Analysis.loc[0, 'price']

df.loc[0:,'Cash Flow Approximation (Hold)'] =df.loc[0,'Net Rent Approximation'] - df.loc[0,'Loan Payment'] 

if D_Payment == 5:
    df.loc[1:,'Cash Flow Approximation (Hold)'] = df['Net Rent Approximation'] - Deep_Analysis.loc[0,'estimate_mortgage']
    
if D_Payment == 20:
    df.loc[1:,'Cash Flow Approximation (Hold)'] = df['Net Rent Approximation'] - Deep_Analysis.loc[0,'estimate mortgage @20%']

df.loc[0,'Cumulative Net Income (Hold)'] = df.loc[0,'Cash Flow Approximation (Hold)']

for i in range(1, len(df)):
    df.at[i, 'Cumulative Net Income (Hold)'] = df.at[i-1, 'Cumulative Net Income (Hold)'] + df.at[i,'Cash Flow Approximation (Hold)']
    

df['MOIC (Hold)'] = -(df.loc[1:, 'Cash Flow Approximation (Hold)'].cumsum() / df.loc[0, 'Cash Flow Approximation (Hold)'])
#df['MOIC (Hold)'] = df['MOIC (Hold)'] * 100

for i in range(0, CashOut) :
    df.at[i, 'Cash Flow Approximation (Sell)'] = df.at[i, 'Cash Flow Approximation (Hold)'] 
    
df.loc[CashOut,'Cash Flow Approximation (Sell)']  = df.loc[CashOut,'Cumulative Principal (Appreciation)']

df.loc[0,'Cumulative Net Income (Sell)'] = df.loc[0,'Cash Flow Approximation (Sell)'] 

for i in range(1, len(df)):
    df.at[i, 'Cumulative Net Income (Sell)'] = df.at[i-1, 'Cumulative Net Income (Sell)'] + df.at[i,'Cash Flow Approximation (Sell)']
    
df['MOIC (Sell)'] = -(df.loc[1:, 'Cash Flow Approximation (Sell)'].cumsum() / df.loc[0, 'Cash Flow Approximation (Sell)'])
#df['MOIC (Sell)'] = df['MOIC (Sell)'] * 100

dfPred = df

#########################################################################################################################
RentInc = 2.5
RentInc = RentInc/100.0
HPI = 0

df = pd.DataFrame(index=loan_months)
df['Loan Months'] = loan_months
df['Loan Payment'] = 0.0
df['Loan Interest'] = 0.0
df['Loan Principal'] = 0.0
df['Loan Balance'] = 0.0
df['Cumulative Principal (No Appreciation)'] = 0.0
df['Cumulative Principal (Appreciation)'] = 0.0
df['Expense Index'] = 0.0
df['Rental Index'] = 0.0
df['Rent Value'] = 0.0
df['Net Rent Approximation'] = 0.0
df['Property Level HPI'] = 0.0
df['Home Value'] = 0.0
df['Cash Flow Approximation (Hold)'] = 0.0
df['Cumulative Net Income (Hold)'] = 0.0
df['MOIC (Hold)'] =0.0
df['Cash Flow Approximation (Sell)'] =0.0
df['Cumulative Net Income (Sell)'] =0.0
df['MOIC (Sell)'] =0.0

df.loc[0, 'Loan Balance'] = (1-(D_Payment/100)) * Deep_Analysis['price'].iloc[0]
df.loc[0, 'Loan Payment'] = Deep_Analysis['price'].iloc[0]*(D_Payment/100)
df.loc[0, 'Loan Principal'] = df.loc[0, 'Loan Payment']
df.loc[1:, 'Loan Payment'] =  (npf.pmt((Int_Rate/100)/12, Term_Months-1, -(df.loc[0, 'Loan Balance'])))
df.loc[1:,'Loan Interest'] = npf.ipmt((Int_Rate/100)/12, df['Loan Months'].iloc[1:], Term_Months-1, -(df.loc[0, 'Loan Balance']))
df.loc[1:,'Loan Principal'] = npf.ppmt((Int_Rate/100)/12, df['Loan Months'].iloc[1:], Term_Months-1, -(df.loc[0, 'Loan Balance']))


for i in range(1, len(df)-1):
    df.at[i, 'Loan Balance'] = df.at[i-1, 'Loan Balance'] - (df.at[i, 'Loan Principal'])
    
    
df.loc[0, 'Cumulative Principal (No Appreciation)'] = df.loc[0, 'Loan Principal']

for i in range(1, len(df)):
    df.at[i, 'Cumulative Principal (No Appreciation)'] = df.at[i-1, 'Cumulative Principal (No Appreciation)'] + (df.at[i, 'Loan Principal'])
    
df.loc[0, 'Expense Index'] = 100
for i in range(1, len(df)):
    df.at[i, 'Expense Index'] = df.at[i-1, 'Expense Index'] * (ExpenseInc + 1) ** (1/12)
    
    
df.loc[0, 'Rental Index'] = 100
for i in range(1, len(df)):
    df.at[i, 'Rental Index'] = df.at[i-1, 'Rental Index'] * (RentInc + 1) ** (1/12)

#df.loc[1, 'Rent Value'] = Deep_Analysis.loc[0, 'Rent est Low'] * df.loc[1, 'Rental Index']
for i in range(1, len(df)):
    df.at[i, 'Rent Value'] = Deep_Analysis.loc[0, 'Rent est Low'] * (df.loc[i, 'Rental Index']/100)

df['Net Rent Approximation'] = df['Rent Value'] * (1-Vacancy)

df.loc[0, 'Property Level HPI'] = 1
for i in range(1, len(df)):
    df.at[i, 'Property Level HPI'] = df.at[i-1, 'Property Level HPI'] * ((HPI/100) + 1) ** (1/12)
    
df['Home Value'] =  Deep_Analysis.loc[0, 'price'] * df['Property Level HPI']

df['Cumulative Principal (Appreciation)'] = df['Cumulative Principal (No Appreciation)'] + df['Home Value'] - Deep_Analysis.loc[0, 'price']

df.loc[0:,'Cash Flow Approximation (Hold)'] =df.loc[0,'Net Rent Approximation'] - df.loc[0,'Loan Payment'] 

if D_Payment == 5:
    df.loc[1:,'Cash Flow Approximation (Hold)'] = df['Net Rent Approximation'] - Deep_Analysis.loc[0,'estimate_mortgage']
    
if D_Payment == 20:
    df.loc[1:,'Cash Flow Approximation (Hold)'] = df['Net Rent Approximation'] - Deep_Analysis.loc[0,'estimate mortgage @20%']

df.loc[0,'Cumulative Net Income (Hold)'] = df.loc[0,'Cash Flow Approximation (Hold)']

for i in range(1, len(df)):
    df.at[i, 'Cumulative Net Income (Hold)'] = df.at[i-1, 'Cumulative Net Income (Hold)'] + df.at[i,'Cash Flow Approximation (Hold)']
    

df['MOIC (Hold)'] = -(df.loc[1:, 'Cash Flow Approximation (Hold)'].cumsum() / df.loc[0, 'Cash Flow Approximation (Hold)'])
#df['MOIC (Hold)'] = df['MOIC (Hold)'] * 100

for i in range(0, CashOut) :
    df.at[i, 'Cash Flow Approximation (Sell)'] = df.at[i, 'Cash Flow Approximation (Hold)'] 
    
df.loc[CashOut,'Cash Flow Approximation (Sell)']  = df.loc[CashOut,'Cumulative Principal (Appreciation)']

df.loc[0,'Cumulative Net Income (Sell)'] = df.loc[0,'Cash Flow Approximation (Sell)'] 

for i in range(1, len(df)):
    df.at[i, 'Cumulative Net Income (Sell)'] = df.at[i-1, 'Cumulative Net Income (Sell)'] + df.at[i,'Cash Flow Approximation (Sell)']
    
df['MOIC (Sell)'] = -(df.loc[1:, 'Cash Flow Approximation (Sell)'].cumsum() / df.loc[0, 'Cash Flow Approximation (Sell)'])
#df['MOIC (Sell)'] = df['MOIC (Sell)'] * 100

dfBear = df


##########################################################################################################################

RentInc = 3.0
RentInc = RentInc/100.0
HPI = 3.5

df = pd.DataFrame(index=loan_months)
df['Loan Months'] = loan_months
df['Loan Payment'] = 0.0
df['Loan Interest'] = 0.0
df['Loan Principal'] = 0.0
df['Loan Balance'] = 0.0
df['Cumulative Principal (No Appreciation)'] = 0.0
df['Cumulative Principal (Appreciation)'] = 0.0
df['Expense Index'] = 0.0
df['Rental Index'] = 0.0
df['Rent Value'] = 0.0
df['Net Rent Approximation'] = 0.0
df['Property Level HPI'] = 0.0
df['Home Value'] = 0.0
df['Cash Flow Approximation (Hold)'] = 0.0
df['Cumulative Net Income (Hold)'] = 0.0
df['MOIC (Hold)'] =0.0
df['Cash Flow Approximation (Sell)'] =0.0
df['Cumulative Net Income (Sell)'] =0.0
df['MOIC (Sell)'] =0.0

df.loc[0, 'Loan Balance'] = (1-(D_Payment/100)) * Deep_Analysis['price'].iloc[0]
df.loc[0, 'Loan Payment'] = Deep_Analysis['price'].iloc[0]*(D_Payment/100)
df.loc[0, 'Loan Principal'] = df.loc[0, 'Loan Payment']
df.loc[1:, 'Loan Payment'] =  (npf.pmt((Int_Rate/100)/12, Term_Months-1, -(df.loc[0, 'Loan Balance'])))
df.loc[1:,'Loan Interest'] = npf.ipmt((Int_Rate/100)/12, df['Loan Months'].iloc[1:], Term_Months-1, -(df.loc[0, 'Loan Balance']))
df.loc[1:,'Loan Principal'] = npf.ppmt((Int_Rate/100)/12, df['Loan Months'].iloc[1:], Term_Months-1, -(df.loc[0, 'Loan Balance']))


for i in range(1, len(df)-1):
    df.at[i, 'Loan Balance'] = df.at[i-1, 'Loan Balance'] - (df.at[i, 'Loan Principal'])
    
    
df.loc[0, 'Cumulative Principal (No Appreciation)'] = df.loc[0, 'Loan Principal']

for i in range(1, len(df)):
    df.at[i, 'Cumulative Principal (No Appreciation)'] = df.at[i-1, 'Cumulative Principal (No Appreciation)'] + (df.at[i, 'Loan Principal'])
    
df.loc[0, 'Expense Index'] = 100
for i in range(1, len(df)):
    df.at[i, 'Expense Index'] = df.at[i-1, 'Expense Index'] * (ExpenseInc + 1) ** (1/12)
    
    
df.loc[0, 'Rental Index'] = 100
for i in range(1, len(df)):
    df.at[i, 'Rental Index'] = df.at[i-1, 'Rental Index'] * (RentInc + 1) ** (1/12)

#df.loc[1, 'Rent Value'] = Deep_Analysis.loc[0, 'Rent est Low'] * df.loc[1, 'Rental Index']
for i in range(1, len(df)):
    df.at[i, 'Rent Value'] = Deep_Analysis.loc[0, 'Rent est Low'] * (df.loc[i, 'Rental Index']/100)

df['Net Rent Approximation'] = df['Rent Value'] * (1-Vacancy)

df.loc[0, 'Property Level HPI'] = 1
for i in range(1, len(df)):
    df.at[i, 'Property Level HPI'] = df.at[i-1, 'Property Level HPI'] * ((HPI/100) + 1) ** (1/12)
    
df['Home Value'] =  Deep_Analysis.loc[0, 'price'] * df['Property Level HPI']

df['Cumulative Principal (Appreciation)'] = df['Cumulative Principal (No Appreciation)'] + df['Home Value'] - Deep_Analysis.loc[0, 'price']

df.loc[0:,'Cash Flow Approximation (Hold)'] =df.loc[0,'Net Rent Approximation'] - df.loc[0,'Loan Payment'] 

if D_Payment == 5:
    df.loc[1:,'Cash Flow Approximation (Hold)'] = df['Net Rent Approximation'] - Deep_Analysis.loc[0,'estimate_mortgage']
    
if D_Payment == 20:
    df.loc[1:,'Cash Flow Approximation (Hold)'] = df['Net Rent Approximation'] - Deep_Analysis.loc[0,'estimate mortgage @20%']

df.loc[0,'Cumulative Net Income (Hold)'] = df.loc[0,'Cash Flow Approximation (Hold)']

for i in range(1, len(df)):
    df.at[i, 'Cumulative Net Income (Hold)'] = df.at[i-1, 'Cumulative Net Income (Hold)'] + df.at[i,'Cash Flow Approximation (Hold)']
    

df['MOIC (Hold)'] = -(df.loc[1:, 'Cash Flow Approximation (Hold)'].cumsum() / df.loc[0, 'Cash Flow Approximation (Hold)'])
#df['MOIC (Hold)'] = df['MOIC (Hold)'] * 100

for i in range(0, CashOut) :
    df.at[i, 'Cash Flow Approximation (Sell)'] = df.at[i, 'Cash Flow Approximation (Hold)'] 
    
df.loc[CashOut,'Cash Flow Approximation (Sell)']  = df.loc[CashOut,'Cumulative Principal (Appreciation)']

df.loc[0,'Cumulative Net Income (Sell)'] = df.loc[0,'Cash Flow Approximation (Sell)'] 

for i in range(1, len(df)):
    df.at[i, 'Cumulative Net Income (Sell)'] = df.at[i-1, 'Cumulative Net Income (Sell)'] + df.at[i,'Cash Flow Approximation (Sell)']
    
df['MOIC (Sell)'] = -(df.loc[1:, 'Cash Flow Approximation (Sell)'].cumsum() / df.loc[0, 'Cash Flow Approximation (Sell)'])
#df['MOIC (Sell)'] = df['MOIC (Sell)'] * 100

dfBase = df


################################################################################################################################
RentInc = 5.0
RentInc = RentInc/100.0
HPI = 5.0

df = pd.DataFrame(index=loan_months)
df['Loan Months'] = loan_months
df['Loan Payment'] = 0.0
df['Loan Interest'] = 0.0
df['Loan Principal'] = 0.0
df['Loan Balance'] = 0.0
df['Cumulative Principal (No Appreciation)'] = 0.0
df['Cumulative Principal (Appreciation)'] = 0.0
df['Expense Index'] = 0.0
df['Rental Index'] = 0.0
df['Rent Value'] = 0.0
df['Net Rent Approximation'] = 0.0
df['Property Level HPI'] = 0.0
df['Home Value'] = 0.0
df['Cash Flow Approximation (Hold)'] = 0.0
df['Cumulative Net Income (Hold)'] = 0.0
df['MOIC (Hold)'] =0.0
df['Cash Flow Approximation (Sell)'] =0.0
df['Cumulative Net Income (Sell)'] =0.0
df['MOIC (Sell)'] =0.0

df.loc[0, 'Loan Balance'] = (1-(D_Payment/100)) * Deep_Analysis['price'].iloc[0]
df.loc[0, 'Loan Payment'] = Deep_Analysis['price'].iloc[0]*(D_Payment/100)
df.loc[0, 'Loan Principal'] = df.loc[0, 'Loan Payment']
df.loc[1:, 'Loan Payment'] =  (npf.pmt((Int_Rate/100)/12, Term_Months-1, -(df.loc[0, 'Loan Balance'])))
df.loc[1:,'Loan Interest'] = npf.ipmt((Int_Rate/100)/12, df['Loan Months'].iloc[1:], Term_Months-1, -(df.loc[0, 'Loan Balance']))
df.loc[1:,'Loan Principal'] = npf.ppmt((Int_Rate/100)/12, df['Loan Months'].iloc[1:], Term_Months-1, -(df.loc[0, 'Loan Balance']))


for i in range(1, len(df)-1):
    df.at[i, 'Loan Balance'] = df.at[i-1, 'Loan Balance'] - (df.at[i, 'Loan Principal'])
    
    
df.loc[0, 'Cumulative Principal (No Appreciation)'] = df.loc[0, 'Loan Principal']

for i in range(1, len(df)):
    df.at[i, 'Cumulative Principal (No Appreciation)'] = df.at[i-1, 'Cumulative Principal (No Appreciation)'] + (df.at[i, 'Loan Principal'])
    
df.loc[0, 'Expense Index'] = 100
for i in range(1, len(df)):
    df.at[i, 'Expense Index'] = df.at[i-1, 'Expense Index'] * (ExpenseInc + 1) ** (1/12)
    
    
df.loc[0, 'Rental Index'] = 100
for i in range(1, len(df)):
    df.at[i, 'Rental Index'] = df.at[i-1, 'Rental Index'] * (RentInc + 1) ** (1/12)

#df.loc[1, 'Rent Value'] = Deep_Analysis.loc[0, 'Rent est Low'] * df.loc[1, 'Rental Index']
for i in range(1, len(df)):
    df.at[i, 'Rent Value'] = Deep_Analysis.loc[0, 'Rent est Low'] * (df.loc[i, 'Rental Index']/100)

df['Net Rent Approximation'] = df['Rent Value'] * (1-Vacancy)

df.loc[0, 'Property Level HPI'] = 1
for i in range(1, len(df)):
    df.at[i, 'Property Level HPI'] = df.at[i-1, 'Property Level HPI'] * ((HPI/100) + 1) ** (1/12)
    
df['Home Value'] =  Deep_Analysis.loc[0, 'price'] * df['Property Level HPI']

df['Cumulative Principal (Appreciation)'] = df['Cumulative Principal (No Appreciation)'] + df['Home Value'] - Deep_Analysis.loc[0, 'price']

df.loc[0:,'Cash Flow Approximation (Hold)'] =df.loc[0,'Net Rent Approximation'] - df.loc[0,'Loan Payment'] 

if D_Payment == 5:
    df.loc[1:,'Cash Flow Approximation (Hold)'] = df['Net Rent Approximation'] - Deep_Analysis.loc[0,'estimate_mortgage']
    
if D_Payment == 20:
    df.loc[1:,'Cash Flow Approximation (Hold)'] = df['Net Rent Approximation'] - Deep_Analysis.loc[0,'estimate mortgage @20%']

df.loc[0,'Cumulative Net Income (Hold)'] = df.loc[0,'Cash Flow Approximation (Hold)']

for i in range(1, len(df)):
    df.at[i, 'Cumulative Net Income (Hold)'] = df.at[i-1, 'Cumulative Net Income (Hold)'] + df.at[i,'Cash Flow Approximation (Hold)']
    

df['MOIC (Hold)'] = -(df.loc[1:, 'Cash Flow Approximation (Hold)'].cumsum() / df.loc[0, 'Cash Flow Approximation (Hold)'])
#df['MOIC (Hold)'] = df['MOIC (Hold)'] * 100

for i in range(0, CashOut) :
    df.at[i, 'Cash Flow Approximation (Sell)'] = df.at[i, 'Cash Flow Approximation (Hold)'] 
    
df.loc[CashOut,'Cash Flow Approximation (Sell)']  = df.loc[CashOut,'Cumulative Principal (Appreciation)']

df.loc[0,'Cumulative Net Income (Sell)'] = df.loc[0,'Cash Flow Approximation (Sell)'] 

for i in range(1, len(df)):
    df.at[i, 'Cumulative Net Income (Sell)'] = df.at[i-1, 'Cumulative Net Income (Sell)'] + df.at[i,'Cash Flow Approximation (Sell)']
    
df['MOIC (Sell)'] = -(df.loc[1:, 'Cash Flow Approximation (Sell)'].cumsum() / df.loc[0, 'Cash Flow Approximation (Sell)'])
#df['MOIC (Sell)'] = df['MOIC (Sell)'] * 100

dfBull = df
    
#############################################################################################################################
    
plt.figure(figsize=(15, 6))

plt.plot(dfPred['Loan Months'], dfPred['Home Value'], label='Predicted Growth')

# Plotting dfBear
plt.plot(dfBear['Loan Months'], dfBear['Home Value'], label='Bear')

# Plotting dfBase
plt.plot(dfBase['Loan Months'], dfBase['Home Value'], label='Base')

# Plotting dfBull
plt.plot(dfBull['Loan Months'], dfBull['Home Value'], label='Bull')

# Set labels and title
plt.xlabel('Loan Months')
plt.ylabel('Home Value')
plt.title('Home Value Scenarios')

# Add legend
plt.legend()

formatter = ticker.StrMethodFormatter('${x:,.0f}')
plt.gca().yaxis.set_major_formatter(formatter)

# Show the plot
plt.show()



################################################################################################################

plt.figure(figsize=(15, 6))

plt.plot(dfPred['Loan Months'], dfPred['MOIC (Sell)']*100, label='Predicted Growth')

# Plotting dfBear
plt.plot(dfBear['Loan Months'], dfBear['MOIC (Sell)']*100, label='Bear')

# Plotting dfBase
plt.plot(dfBase['Loan Months'], dfBase['MOIC (Sell)']*100, label='Base')

# Plotting dfBull
plt.plot(dfBull['Loan Months'], dfBull['MOIC (Sell)']*100, label='Bull')

# Set labels and title
plt.xlabel('Loan Months')
plt.ylabel('MOIC')
plt.title('MOIC Sell Scenario')

# Add legend
plt.legend()

formatter = ticker.StrMethodFormatter('%{x:,.0f}')
plt.gca().yaxis.set_major_formatter(formatter)

# Show the plot
plt.show()

##################################################################################################################

plt.figure(figsize=(15, 6))

plt.plot(dfPred['Loan Months'], dfPred['MOIC (Hold)']*100, label='Predicted Growthh')

# Plotting dfBear
plt.plot(dfBear['Loan Months'], dfBear['MOIC (Hold)']*100, label='Bear')

# Plotting dfBase
plt.plot(dfBase['Loan Months'], dfBase['MOIC (Hold)']*100, label='Base')

# Plotting dfBull
plt.plot(dfBull['Loan Months'], dfBull['MOIC (Hold)']*100, label='Bull')

# Set labels and title
plt.xlabel('Loan Months')
plt.ylabel('MOIC')
plt.title('MOIC Hold Scenario')

# Add legend
plt.legend()

formatter = ticker.StrMethodFormatter('%{x:,.0f}')
plt.gca().yaxis.set_major_formatter(formatter)

# Show the plot
plt.show()

####################################################################################################



####################################################################################################

zipcode = pd.read_csv('Zip_zhvi_uc_sfrcondo_tier_0.33_0.67_sm_sa_month.csv')


AddressWCode = Deep_Analysis.iloc[0, 0]
ZipCode = int(AddressWCode[-5:])


new_dataframe = zipcode[zipcode['RegionName'] == ZipCode].copy()

new_dataframe = new_dataframe.loc[:, '2000-01-31':]

new_dataframe.reset_index(drop=True, inplace=True)


transposed_dataframe = new_dataframe.transpose()


transposed_dataframe.index = pd.to_datetime(transposed_dataframe.index)


transposed_dataframe.columns = transposed_dataframe.columns.astype(str)



ValueTS = transposed_dataframe['0']

plt.figure(figsize=(10, 6))
plt.plot(ValueTS.index, ValueTS.values)
plt.xlabel('Date')
plt.ylabel('Value')
plt.title(Deep_Analysis.iloc[0, 0] + ' Zip Code Value Trend')
plt.grid(True)
plt.show()

```

    What is the current interest rate?
    7
    What percent do you intend to put down? (5 or 20)
    20
    What term length? (15 or 30 years)
    30
    What is the Annual Expense Increase Rate?
    3
    What is the Annual Rent Increase Rate?
    4
    What is the predicted vacancy?
    5
    What is the predicted Annual HPI?
    6
    After how many months do you intend to sell?
    If you don't, enter 0 
    100
                                        address   latitude  longitude     price  \
    0  400 W Ontario St #607, Chicago, IL 60654  41.893356  -87.63896  350000.0   
    
       bathrooms  bedrooms  area_sqft  \
    0        2.0       2.0     1100.0   
    
                                            property_url  estimate_mortgage  \
    0  https://www.zillow.com/homedetails/400-W-Ontar...            3554.58   
    
       Rent est Low  Rent est Avg  Cash Flow Low 5%  Cash Flow Avg 5%  \
    0        3300.0      3574.875           -254.58            20.295   
    
       Cash Flow Low 20%  Cash Flow Avg 20%  estimate mortgage @20%  
    0              284.0            558.875                  3016.0  
    


    
![png](output_23_1.png)
    



    
![png](output_23_2.png)
    



    
![png](output_23_3.png)
    



    
![png](output_23_4.png)
    



    
![png](output_23_5.png)
    


# Further Imporvements

Soon
- Incorporate other financial models/mathematical approaches for further insight into investments
- Further develop machine learning applications to improve model accuracy/fit
- Incorporate zip code value trends into property value projections
- Make geospatial maps interactive (can click on dots that give links to properties)
- Add inputs for potential variables like property renovations


Far
- Utilize image recognition AI to grade property conditions and incorporate it into ML models
- Incorporate other buy/rent databases such as apartments.com
- Build out the front end for more fluid interactability
- Develop an option for commercial investments with similar logic
