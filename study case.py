#Importing data
import pandas as pd
import numpy as np
import ipywidgets as widgets
from matplotlib import gridspec
from matplotlib import pyplot as plt
base = pd.read_csv('Data Analyst CAse Study_May 2021.csv')
print(base.head())

base.dtypes

#Detecting missing values
pd.isnull(base).sum()



#excluding attributes
base.drop(['Description'],1,inplace=True)
print(base.head())

#Filling up missing values
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values = np.nan, strategy = 'most_frequent')
base['Current Type'] = imputer.fit_transform(base[['Current Type']])

from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values = np.nan, strategy = 'most_frequent')
base['Current Nurturing'] = imputer.fit_transform(base[['Current Nurturing']])

pd.isnull(base).sum()


#Creating insights

base[['Event', 'Name']].groupby('Name').count().reset_index()

base[['Name','Event']].groupby('Event').count().reset_index()

base[['Name','Current Type']].groupby('Current Type').count().reset_index()

base[['Event','Current Nurturing']].groupby('Current Nurturing').count().reset_index()

base[['Event', 'Name', 'Date']].groupby('Date').count().reset_index()

event_per_person = base[['Event', 'Name']].groupby('Name')

person_per_event = base[['Name','Event']].groupby('Event')

person_event_date = base[['Event', 'Name', 'Date']].groupby('Date')

for Name, frame in event_per_person:
    print (frame.head(), end='\n\n')
    
for Date, frame in person_event_date:
    print (frame.head(), end='\n\n')
    
    
base[['Event', 'Name']][base['Name'] == 'Aaron Banks']

base[['Event', 'Name']][base['Event'] == 'Lead Pipeline Update']

# Grafic Visualization

# For that we change and create new date format, such as year, month and year-week

base['year'] = pd.to_datetime(base['Date']).dt.strftime('%Y')
base['month'] = pd.to_datetime(base['Date']).dt.strftime('%m')
base['year_week'] = pd.to_datetime(base['Date']).dt.strftime('%Y-%U')

# Widgets to control data

Name_limit = widgets.Dropdown(
    options=base['Name'].sort_values().unique().tolist(),
    value = 'Jesse Drennon',
    description = 'Name',
    disable = False)

Event_limit = widgets.Dropdown(
    options=base['Event'].sort_values().unique().tolist(),
    value = 'Lead Created',
    description = 'Name',
    disable = False)

def update_map(base,e_limit,n_limit):
    df = base[(base['Name'] == n_limit) &
              (base['Event'] == e_limit)]
    
    fig = plt.figure(figsize=(21, 12))
    specs = gridspec.GridSpec(ncols=2, nrows=4, figure=fig)

    ax1 = fig.add_subplot(specs[0, :])  # First Rows
    ax2 = fig.add_subplot(specs[1, 0])  # Second Row -  First Column
    ax3 = fig.add_subplot(specs[1, 1])  # Second Row -  Second Column
    ax4 = fig.add_subplot(specs[2, :])  # Third Row -  First Column

    by_year = df[['Event', 'year']].groupby('year').count().reset_index()
    ax1.bar(by_year['year'], by_year['Event'])
    ax1.set_title('title: Number of Event by year')

    by_day = df[['Event', 'Date']].groupby('Date').count().reset_index()
    ax2.bar(by_day['Date'], by_day['Event'])
    ax2.set_title('title: Number of Event by day')

    by_week_of_year = df[['Event', 'year_week']].groupby('year_week').count().reset_index()
    ax3.bar(by_week_of_year['year_week'], by_week_of_year['Event'])
    ax3.set_title('title: Number of Event by week of year')

    by_month = df[['Event', 'month']].groupby('month').count().reset_index()
    ax3.bar(by_month['month'], by_month['Event'])
    ax3.set_title('title: Number of Event by month')
    
widgets.interactive(update_map, base=(base), e_limit = Event_limit, n_limit = Name_limit)
