#!/usr/bin/env python
# coding: utf-8

# # Import Libraries

# In[1]:



import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import gmplot 
import warnings

warnings.filterwarnings('ignore')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


# Set Directory

Dir = '/Users/adamchappell/Documents/Monroe County Analysis/'


# In[3]:


#Set Pandas Display options

pd.set_option('display.max_columns', None)
pd.set_option('display.min_rows', 50)
pd.set_option('expand_frame_repr', False)

# pd.describe_option()


# In[4]:


#Read In the Data

crashes = pd.read_csv((Dir + 'monroe-county-crash-data2003-to-2015.csv'), encoding = "ISO-8859-1")


# # Examine the Data

# In[5]:


# Run Summary Statistics


print("\nshape:\n", crashes.shape)


crashes.describe()


# Something looks funny in the latitude and Longitude. min is 0 for latitude even though most are around 39. 
# Longitudes should all be around -86, but the max value is positive.
# 
# I'll bet there was some inputted information here. Luckily, this won't be too consequential for the analysis we are doing. We aren't worried about coordinates until we are map plotting.
# 
# However, at some point we should probably count the invalid coordinates (0's, positives where it should be negative, etc. )
# 

# In[6]:


#Let's look at the number of total values, unique values, and null values per column

print("\nNumber of values by column\n\n", crashes.count())

#count unique values

print("\nNumber of unique values by column\n\n", crashes.nunique())

print("\nNumber of nulls by column\n\n", crashes.isnull().sum())







# Notice Latitude, Longitude, and Reported Location don't always match up
# 
# there are over 1000 missing values for primary factor (about 2%), and more than a few hours, reported locations, and other data missing.  Overall, though, when compared to complete records they are relatively few. 
# 
# 

# Since these are mostly categorical variables we are dealing with here, 
# it would be a good idea to note the categories and the number of records in each one. 

# # Value Counts by Different Categories
# 

# In[7]:


#counts the number of crashes in each unique category. Also sorts them in descending order. 


pd.set_option('display.min_rows', 20)

for col in crashes.columns:
    if col == "Master Record Number":
        pass
        
    else:
        print(col, "  Value Counts\n", crashes[col].value_counts(), "\n")


# Note: day 1 = Sunday, day 2 = Monday, day 7 = Saturday, etc (Based on looking at the 'Weekend' variable). Might be helpful to Change the numbers to strings at some point, maybe.
# 
# This gives us a good overview of the data in terms of what it can describe.
# 
# There are some categories that I find problematic, there is ambiguity around what they represent. For Example, in "Injury Type", "No injury/Unkown" is by far the largest category. But the "Unkown" in the label makes me wonder if some of these were in fact unreported injuries. Additionally, in Primary Factor, There are several categories that simpy include as part of it, "Explain in narrative"-- this represents significant missing data. 
# 
# It might be interesting at some point to look at the subsets of data that have these ambiguous categories and see how it is distributed.
# 
# From the Location list we can easily see what the top crash sites are. They are E 3rd St, W 3rd St (Probably on the same road), and SR37N & VERNAL
# 
# Caveat: The site mentions people may have put in different names for the same street. I'm not sure how to fix this easily. 

# # Time Trends

# In[8]:


# Make a gradient pivot table of crashes across time. 

piv_year_month = pd.pivot_table(crashes, values="Master Record Number", index="Month", columns="Year", aggfunc='count', fill_value=None, margins=False, dropna=True, margins_name='All', observed=False)

cm = sns.light_palette("green", as_cmap=True)

r = piv_year_month.style.background_gradient(cmap=cm)
r


# Suprisingly, Most crashes occur in the fall and going into winter. I thought most occured during the summer when people went on vacation. 

# Let's Look at how crashes and fatalities have gone up or down over time

# In[9]:



#this pivot table includes all crash types
piv_year_all = pd.pivot_table(crashes, values="Master Record Number", index="Year", aggfunc='count', fill_value=None, margins=False, dropna=True, margins_name='All', observed=False)
piv_year_all = piv_year_all.rename(columns={'Master Record Number': 'Crashes'})

#this one narrows it to fatalities. 
piv_year_fatal = pd.pivot_table(crashes[crashes["Injury Type"]=='Fatal'], values="Master Record Number", index="Year", aggfunc='count', fill_value=None, margins=False, dropna=True, margins_name='All', observed=False)
piv_year_fatal = piv_year_fatal.rename(columns={'Master Record Number': 'Fatalities'})


#this is simply putting them together and graphing them
piv_year = pd.concat([piv_year_all, piv_year_fatal], axis=1)

                                                
piv_year.plot.bar(subplots=True)


# Doesn't seem like there has been a huge trend up or down for either. 

# Ok, We're done with across time for now. Let's check out a typical week and how crashes are distrubuted there

# In[10]:


# Pivot Table-- # Collisions By Hour and Day

piv_day_hour = pd.pivot_table(crashes, values="Master Record Number", index="Hour", columns="Day", aggfunc='count', fill_value=None, margins=False, dropna=True, margins_name='All', observed=False)


s = piv_day_hour.style.background_gradient(cmap=cm)
s


# The darker squares represent higher values.  Notice most crashes occur around 4 or 5 pm, except on Saturday and sunday, When both midday and nightime crashes increase somewhat. Makes sense.

# # Location Analysis
# 

# In[11]:


#Compare Unique Locations to day of the week, look for correlations
piv_location_day = pd.pivot_table(crashes, 
                              values="Master Record Number", 
                              index="Reported_Location", 
                              columns=("Day"), 
                              aggfunc='count', 
                              fill_value=None, margins=True, dropna=True, margins_name='All', observed=False)

piv_location_day.sort_values(by=('All'), ascending=False,inplace=True)

#Drop " 'All' Values"
piv_location_day = piv_location_day.drop("All", axis=0)
piv_location_day = piv_location_day.drop("All", axis=1)

#plot gradient pivottable (top 50 locations only)
t = piv_location_day[:50].style.background_gradient(cmap=cm)
t


# It seems the top five streets for crashes  happen mostly during the workweek. the Next two streets (E. 10th and N Walnut) seem to be weekend crash sites. 

# In[12]:


#Note: This Cell was causing the Kernel to stop for some reason so I have commented it out for now. Should work individually though. 

# #Compare Streets to hour, look for correlations (Not that useful)
# piv_location_hour = pd.pivot_table(crashes, 
#                               values="Master Record Number", 
#                               index="Reported_Location", 
#                               columns=("Hour"), 
#                               aggfunc='count', 
#                               fill_value=None, margins=True, dropna=True, margins_name='All', observed=False)

# #sort by crash count
# piv_location_hour.sort_values(by=('All'), ascending=False,inplace=True)

# print(piv_location_hour.shape[0])

# #Drop " 'All' Values"
# piv_location_hour = piv_location_hour.drop("All", axis=0)
# piv_location_hour = piv_location_hour.drop("All", axis=1)


# cm = sns.light_palette("green", as_cmap=True)
# t = piv_location_hour[:100].style.background_gradient(cmap=cm)
# t


# Top crash sites have the most crashes during the day. Consistent with earlier Day/Time Heatmap
# 
# Let's check out if there are any particularly deadly or incapacitating roads

# In[13]:



# sort locations by crash counts and get top 10 sites
locations = crashes['Reported_Location'].value_counts()
top_locations = locations[:10]


#pivot locations by injury type
piv_location_crash = pd.pivot_table(crashes, 
                              values="Master Record Number", 
                              index="Reported_Location", 
                              columns=("Injury Type"), 
                              aggfunc='count', 
                              fill_value=None, margins=True, dropna=True, margins_name='All', observed=False)

# sortby = 'Incapacitating
sortby = 'Fatal'

piv_location_crash.sort_values(by=(sortby), ascending=False,inplace=True)


print(piv_location_crash)

# #Drop " 'All' Values"
# piv_location = piv_location.drop("All", axis=0)
# piv_location = piv_location.drop("All", axis=1)

# t = piv_location_crash.style.background_gradient(cmap=cm)
# t


# There doesn't seem to be a particularly deadly or frequently incapacitating street to have an accident on (sorted streets by fatalities and then by # of incapacitating injuries.)
# 
# Let's see what type of collisions are deadliest

# In[14]:


# pivot of collision type to injury type

# hypothesis: pedestrian, bicycle collisions will have higher fatalities/incapacitations than car collision

piv_type = pd.pivot_table(crashes, values="Master Record Number", index="Collision Type", columns=("Injury Type"),
                          aggfunc='count', fill_value=None, margins=True, dropna=True, margins_name='All', observed=False)

print()

piv_type.sort_values(by=('All'), ascending=False,inplace=True)

print(piv_type)


# In[15]:


# So what Factors are leading  to the top three collision types for fatalities?


#Pivot table only includes factors that contributed to 1-car, 2-car, or motorcycle crashes
piv_factor = pd.pivot_table(crashes[crashes["Collision Type"].isin(("1-Car", "2-Car", "Moped/Motorcycle"))] , 
                          values="Master Record Number", index=("Primary Factor"), columns=("Injury Type"),
                          aggfunc='count', fill_value=None, margins=False, dropna=True, margins_name='All', observed=False)


piv_factor.sort_values(by=('Fatal'), ascending=False,inplace=True)

#print top 10
print(piv_factor[:10])



# Running off the road, driving left of center, and unsafe speeds are deadly factors. Running off the road is not one I was expecting. 

# Below are some Extra analysis before we get to the final map plotter. 

# In[16]:


# Zero in on one-car collisions


piv_factor = pd.pivot_table(crashes[crashes["Collision Type"].isin(("1-Car", ))] , 
                          values="Master Record Number", index=("Primary Factor"), columns=("Injury Type"),
                          aggfunc='count', fill_value=None, margins=False, dropna=True, margins_name='All', observed=False)

print()

piv_factor.sort_values(by=('Fatal'), ascending=False,inplace=True)

print(piv_factor[:10])


# In one-car crashes, Running off the road, unsafe speeds, and alchohol seem to be involved the most. 

# In[17]:


#ran off the road... but what roads??
piv_factor_roads = pd.pivot_table(crashes[(crashes["Collision Type"].isin(("1-Car", )) )& 
                                          (crashes["Primary Factor"].isin(("RAN OFF ROAD RIGHT", )) )] , 
                          values="Master Record Number", index=("Primary Factor", "Reported_Location"), columns=("Injury Type"),
                          aggfunc='count', fill_value=None, margins=True, dropna=True, margins_name='All', observed=False)

print()

piv_factor_roads.sort_values(by=('All'), ascending=False,inplace=True)

pd.set_option('display.min_rows', 100)
print(piv_factor_roads[:20])


# Running off the road occurs on some roads more than others (17th street & Monroe is a primary offender), but there isn't any outlier that I was hoping to catch

# In[18]:


#Zero in on two-car collisions

# for a later time


# In[19]:


#Zero in on motorcycle collisions

# for a later time


# # Alcohol influence
# 

# In[20]:


#Let's see how a day of the week/ time of day count changes when we limit the crashes to those involving alcohol

piv_alcohol_day_hour = pd.pivot_table(crashes[(crashes["Primary Factor"]== "ALCOHOLIC BEVERAGES")], values="Master Record Number", index="Hour", columns="Day", aggfunc='count', fill_value=None, margins=False, dropna=True, margins_name='All', observed=False)


s = piv_alcohol_day_hour.style.background_gradient(cmap=cm)
s


# More Alcohol involved accidents occur at night. no surprise there. 

# # How much more likely are pedestrian, bicycle, and motorcycle crashes to be fatal than car-only crashes?

# Hypothesis: Though they occur less often, they are more likely to be fatal by an order of magnitude, at least. 

# In[21]:


#maps collision type to fatalities
piv_col_type_fatal = pd.pivot_table(crashes[crashes["Injury Type"]=="Fatal"], values="Master Record Number", index="Collision Type", aggfunc='count', fill_value=None, margins=False, dropna=True, margins_name='All', observed=False)
piv_col_type_fatal = piv_col_type_fatal.rename(columns={'Master Record Number': 'Fatalities'})

print(piv_col_type_fatal)

#maps collision type to incapacitation
piv_col_type_inc = pd.pivot_table(crashes[crashes["Injury Type"]=='Incapacitating'], values="Master Record Number", index="Collision Type", aggfunc='count', fill_value=None, margins=False, dropna=True, margins_name='All', observed=False)
piv_col_type_inc = piv_col_type_inc.rename(columns={'Master Record Number': 'Incapacitating'})

print(piv_col_type_inc)

#maps collision type to total records
piv_col_type_total = pd.pivot_table(crashes, values="Master Record Number", index="Collision Type", aggfunc='count', fill_value=None, margins=False, dropna=True, margins_name='All', observed=False)
piv_col_type_total = piv_col_type_total.rename(columns={'Master Record Number': 'Total Crashes Reported'})

print(piv_col_type_total)


#joins the three
piv_col_type_all = pd.concat([piv_col_type_fatal, piv_col_type_inc, piv_col_type_total], axis=1)

#creates fatality ratio variable
piv_col_type_all["Percent Fatal"] = piv_col_type_all["Fatalities"] / piv_col_type_all["Total Crashes Reported"]

#creates incapacitating ratio variable
piv_col_type_all["Percent Incapacitating"] = piv_col_type_all["Incapacitating"] / piv_col_type_all["Total Crashes Reported"]

#deletes all columns except ratio variables
piv_col_perc = piv_col_type_all.drop(["Fatalities", "Incapacitating", "Total Crashes Reported"], axis=1)


print(piv_col_perc)
                                                
piv_col_perc.plot.bar(subplots=False)


# I was correct. Notice How Motorcycle Accidents are the most likely to be fatal out of any kind. Also if you crash on a motorcycle, holding all other variables constant, there is a 16% chance you will be seriously injured.  

# # How has distracted driving and cell phone related accidents changed over time?
# 
# 

# 

# In[22]:


# Hypothesis: Increased

piv_cell = pd.pivot_table(crashes[crashes["Primary Factor"]=="CELL PHONE USAGE"], 
                          values="Master Record Number", index=("Year"), 
                          aggfunc='count', fill_value=None, margins=False, dropna=True, margins_name='All', observed=False)

piv_cell = piv_cell.rename(columns={'Master Record Number': 'Cell-phone related crashes'})

print(piv_cell)

piv_dd = pd.pivot_table(crashes[crashes["Primary Factor"]=="DRIVER DISTRACTED - EXPLAIN IN NARRATIVE"], 
                          values="Master Record Number", index=("Year"), 
                          aggfunc='count', fill_value=None, margins=False, dropna=True, margins_name='All', observed=False)

piv_dd = piv_dd.rename(columns={'Master Record Number': 'Distracted-Driver related crashes'})

print(piv_dd)

# piv_year_fatal = piv_year_fatal.rename(columns={'Master Record Number': 'Fatalities'})


piv_ddt = pd.concat([piv_cell, piv_dd], axis=1)

print(piv_ddt)

                                                
piv_ddt.plot.line(subplots=False)
piv_ddt.plot.line(subplots=True)


# Cell Phone and Distracted Driver related crashes have decreased from a high point between 2008-2010, which was about the time smart phones became popular

# # Map

# Let's Map where the top Crash Sites are

# In[23]:


# Google Map

# GoogleMapPlotter return Map object 
# Pass the center latitude and 
# center longitude 

#here we will make an empty map centered in Monroe County
gmap1 = gmplot.GoogleMapPlotter(39.16465511, 
                                -86.533408, 13 ) 
    
# Pass the absolute path 
gmap1.draw( Dir + "map11.html" )

#Now we will make a heatmap with top crash sites by # of occurance

# Create a new subset that includes only crashes in the top ten crash locations
top_loc_crashes =  crashes[crashes["Reported_Location"].isin(top_locations.index)]
    
top_loc_crashes.shape   

# heatmap plot heating Type 
# points on the Google map 
points = 2000
gmap1.heatmap( top_loc_crashes["Latitude"], top_loc_crashes["Longitude"]) 
  
gmap1.draw( Dir + "map14.html" ) 

# find these maps in your project diretory and click to open


# If you look at the heatmap, it looks as though the highest number of accidents occur on highway 46 (3rd street) running east to west and Walnut Street running north to south.
# 
# That makes sense because they go to the heart of the city so they are probably be busiest
# 
# It would be interesting to get a proxy variable for traffic flow and compare traffic flow to crash rate to see which road a person is truly most likely to crash on, controlling for traffic. In other words, where is the road itself worse, not because of who drives on it but because of the road.  
# 
# 

# # Where to go further...

# # # Interesting Bolean Variables to Possibly Add and Analyze

# group hours into general time of day (morning, afternoon, evening, etc. )
# 
# car collision:  1- 2- or 3+ -collisions (and maybe bus?)
# non-car:  pedestrian, bicycle,  motorcycle together
# 
# serious injury: fatal or incapacitated
# 
# Under the influence: Illegal drugs/ Alcohol involved
# 
# distracted: includes cell phone and "distracted" general
# 
# We could group the primary factors together in a number of ways depending on our goal. 

# # # Remove observations with missing data and run analysis again?
# 
# # # Run Predictions for future months/years 

# In[ ]:




