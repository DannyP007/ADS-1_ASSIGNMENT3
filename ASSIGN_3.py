# -*- coding: utf-8 -*-
"""
Created on Thu Jan 19 20:31:37 2023

@author: mauli
"""

# In[]: 
    
# For Clustring

# In[]: 


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import cluster # for Culstering libraries
import sklearn.metrics as skmet
from sklearn.metrics import silhouette_score

# In[]: data Cleaning and read the file from location
    
# read file from location
def readFile(y):
    '''
        This function is used to read csv file in original form and then 
        filling 'nan' values by '0' to avoid disturbance in data visualization
        and final results.

        Parameters
        ----------
        x : csv filename
    
        Returns
        ------- 
        gdp_growth : variable for storing csv file
'''
    
    Inflation = pd.read_csv("Inflation_gdp.csv");
    Inflation = pd.read_csv(y)
    Inflation = Inflation.fillna(0.0)
    return gdp

Inflation = pd.read_csv("Inflation_gdp.csv")
print("\nPopulation: \n", Inflation)


Inflation = Inflation.drop(['Country Code', 'Indicator Name', 'Indicator Code', '1960','1961',	'1962',	'1963','1964','1965','1966','1967','1968','1969','1970','1971','1972',
'1973','1974','1975','1976','1977','1978','1979','1980','1981','1982','1983','1984','1985','1986','1987','1988','1989',], axis=1)
print("\nNew Population: \n", Inflation)

Infaltion = Inflation.fillna(0)
print("\nNew Population after filling null values: \n", Inflation)

Inflation = pd.DataFrame.transpose(Inflation)
print("\nTransposed Dataframe: \n",Inflation)

header = Inflation.iloc[0].values.tolist()
Inflation.columns = header
print("\nPopulation Header: \n",Inflation)

Inflation= Inflation.iloc[2:]
print("\nNew Transposed Dataframe: \n",Inflation)

Inflation_ex = Inflation[["India","Spain"]].copy()

max_val = Inflation_ex.max()
min_val = Inflation_ex.min()
Inflation_ex = (Inflation_ex - min_val) / (max_val - min_val)
print("\nNew selected columns dataframe: \n", Inflation_ex)

# In[]: For culster, make culster
ncluster = 5
kmeans = cluster.KMeans(n_clusters=ncluster)

kmeans.fit(Inflation_ex)

labels = kmeans.labels_

# extract the estimated cluster centres
cen = kmeans.cluster_centers_
print(cen)

# calculate the silhoutte score
print(skmet.silhouette_score(Inflation_ex, labels))

# In[]: for better visualization 

# plot using the labels to select colour
plt.figure(figsize=(10.0, 10.0))
col = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown", \
"tab:pink", "tab:gray", "tab:olive", "tab:cyan"]
    
for l in range(ncluster): # loop over the different labels
    plt.plot(Inflation_ex[labels==l]["India"], Inflation_ex[labels==l]["Spain"], marker="+", markersize=3, color=col[l])    
    
# show cluster centres
for ic in range(ncluster):
    xc, yc = cen[ic,:]  
    plt.plot(xc, yc, "dk", markersize=15)
plt.xlabel("India")
plt.ylabel("Spain")
plt.show()    

print(cen)

df_cen = pd.DataFrame(cen, columns=["India", "Spain"])
print(df_cen)
df_cen = df_cen * (max_val - min_val) + max_val
Inflation_ex = Inflation_ex * (max_val - min_val) + max_val
# print(df_ex.min(), df_ex.max())

print(df_cen)

# In[]: plotting culster

# plot using the labels to select colour
plt.figure(figsize=(10.0, 10.0))

col = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown", \
"tab:pink", "tab:gray", "tab:olive", "tab:cyan"]
for l in range(ncluster): # loop over the different labels
    plt.plot(Inflation_ex[labels==l]["India"], Inflation_ex[labels==l]["Spain"], "+", markersize=3, color=col[l])
    
# show cluster centres
plt.plot(df_cen["India"], df_cen["Spain"], "dk", markersize=15)
plt.xlabel("India")
plt.ylabel("Spain")
plt.title('The Inflation rate of India and Spain')
plt.show()
print(cen)    
plt.savefig("inflation.jpg")

# In[]: -------------------------------------------------------------------

# In[]:
    
# For curve and Fitting

# In[]: IMPORT LIBRARIES FOR CURVE & FIT
    
import numpy as np
import pandas as pd
import scipy.optimize as opt
import matplotlib.pyplot as plt
#import errors as err


def readFile(y):
    '''
        This function is used to read csv file in original form and then 
        filling 'nan' values by '0' to avoid disturbance in data visualization
        and final results.

        Parameters
        ----------
        x : csv filename
    
        Returns
        ------- 
        gdp_growth : variable for storing csv file
'''
    
    gdp = pd.read_csv("gdp.csv");
    gdp = pd.read_csv(y)
    gdp = gdp.fillna(0.0)
    return gdp

# In[]: Data Cleaning

gdp = pd.read_csv("gdp.csv")
print(gdp)

gdp = pd.DataFrame(gdp)

gdp = gdp.transpose()
print("\nGDP: \n", gdp)

# create a header for make good data

header3 = gdp.iloc[0].values.tolist()
gdp.columns = header3
print("\nPopulation Header: \n",gdp)

# access perticular column
gdp = gdp["Spain"]
print("\nGDP after dropping columns: \n", gdp)

gdp.columns = ["GDP"]
print("\nGDP: \n",gdp)
#access through iloc variable
gdp = gdp.iloc[5:]
gdp = gdp.iloc[:-1]
print("\nGDP: \n",gdp)

# set the index
gdp = gdp.reset_index()
print("\nGDP index: \n",gdp)
#rename the column
gdp = gdp.rename(columns={"index": "Year", "Spain": "GDP"} )
print("\nGDP rename: \n",gdp)
#simple plot line
print(gdp.columns)
gdp.plot("Year", "GDP")
plt.show()

# In[]: Use Exponential function for calculates growth rate
def exponential(s, q0, h):
    """Calculates exponential function with scale factor n0 and growth rate g."""
    s = s - 1970.0
    x = q0 * np.exp(h*s)
    return x

print(type(gdp["Year"].iloc[1]))
gdp["Year"] = pd.to_numeric(gdp["Year"])
print("\nGDP Type: \n", type(gdp["Year"].iloc[1]))
param, covar = opt.curve_fit(exponential, gdp["Year"], gdp["GDP"],
p0=(4.978423, 0.03))

# set the data for fit
# In[]: for plotting 

plt.figure()
gdp["fit"] = exponential(gdp["Year"], *param)
gdp.plot("Year", ["GDP", "fit"], label=["New GDP", "New Fit"])
plt.legend()
plt.title("The GDP In US($)")
plt.ylabel("Number In Us($)")
plt.show()
plt.savefig("Gdp.jpg")# save figure 

# In[]: forcasting fit for future
    
year = np.arange(1960, 2030)

print("\nForecast Years: \n", year)

forecast = exponential(year, *param)

plt.figure()
plt.plot(gdp["Year"], gdp["GDP"], label="GDP")
plt.plot(year, forecast, label="Forecast")

plt.xlabel("Year")
plt.ylabel("GDP")
plt.title("GDP Growth")
plt.legend()
plt.show()
# In[]: 
    
def err_ranges(x, exponential, param, sigma):
    """
    Calculates the upper and lower limits for the function, parameters and
    sigmas for single value or array x. Functions values are calculated for 
    all combinations of +/- sigma and the minimum and maximum is determined.
    Can be used for all number of parameters and sigmas >=1.
    
    This routine can be used in assignment programs.
    """
    import itertools as iter
    
    # initiate arrays for lower and upper limits
    lower = exponential(x, *param)
    upper = lower
    
    uplow = []   # list to hold upper and lower limits for parameters
    for p,s in zip(param, sigma):
        pmin = p - s
        pmax = p + s
        uplow.append((pmin, pmax))
        
    pmix = list(iter.product(*uplow))
    
    for p in pmix:
        y = exponential(x, *p)
        lower = np.minimum(lower, y)
        upper = np.maximum(upper, y)
        print("\nLower: \n", lower)
        print("\nUpper: \n", upper)        
    return lower, upper


