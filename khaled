# Covid-19.EDA

from google.colab import drive
drive.mount('/content/drive')

#IMPORTING LIBRARIES

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
import plotly.express as px

#EXPLORING THE DATA

df = pd.read_csv('/content/drive/MyDrive/WHO-COVID-19-global-data.csv')
df.head()

df.info()

df.describe()

pd.options.display.float_format = '{:,.0f}'.format
df.describe()

#DUPLICATE VALUES

df.duplicated().sum()

df.drop_duplicates(inplace=True)

df.duplicated().sum()

#MISSING VALUES

df.isnull().sum()

round(df.isna().mean() * 100 ,2)

df.columns = df.columns.str.lower()

df.sample()

#DATE REPORTED

df.date_reported.dtype

df.date_reported.describe()

df.date_reported.min(), df.date_reported.max()

df['date_reported'] = pd.to_datetime(df['date_reported'], format='%Y-%m-%d')

df['date_reported'].dtype

df['date_reported'].dtype.type

df.info()

df.head()

df.date_reported.max() - df.date_reported.min()

total_period = df.date_reported.max() - df.date_reported.min()
total_years = total_period/ np.timedelta64(1, 'Y')
total_months = total_period/ np.timedelta64(1, 'M')
total_days = total_period/ np.timedelta64(1, 'D')
print(f'Total period: {total_period}')
print(f'Total years: {total_years}')
print(f'Total months: {total_months}')
print(f'Total days: {total_days}')

from dateutil.relativedelta import relativedelta

relativedelta(df.date_reported.max(), df.date_reported.min())

from datetime import datetime
current_date = datetime.now()
current_date


current_date = datetime.today()
current_date

relativedelta(df.date_reported.max(), current_date)

relativedelta(current_date, df.date_reported.max())

COUNTRY CODE AND COUNTRY

df.country_code.nunique(), df.country.nunique()

df.drop('country_code', axis =1, inplace=True)

round(df.isna().mean() * 100 ,2)

df.country.unique()

'Egypt' in df.country.unique()

#FILTER BY COUNTRY

df[df.country == 'Egypt'].shape

plt.figure(figsize=(12,6))
sns.lineplot(x='date_reported', y='cumulative_cases', data=df[df.country == 'Egypt'])
plt.title('Cumulative Cases Over Time')
plt.show()

px.line(df[df.country == 'Egypt'], x='date_reported', y='cumulative_cases')

px.line(df[df.country == 'Egypt'], x='date_reported', y='cumulative_cases',
         title='Cumulative Cases Over Time', width=800, height=400)

def cumulative_cases(country):
    fig = px.line(df[df.country == country], x='date_reported', y='cumulative_cases',
                    title= f'Cumulative Cases Over Time for {country}', width=800, height=400)
    fig.show()

cumulative_cases('Egypt')

cumulative_cases('United States of America')

def cumulative_deaths(country):
    fig = px.line(df[df.country == country], x='date_reported', y='cumulative_deaths',
                     title= f'Cumulative deaths Over Time for {country}', width=800, height=400)
    fig.show()

cumulative_deaths('Egypt')

cumulative_deaths('United States of America')

def cumulative_cases_period(country, start_date, end_date):
    data = df[(df.country == country) & (df.date_reported >= start_date) & (df.date_reported <= end_date)]
    fig = px.line(data, x='date_reported', y='cumulative_cases', title= f'Cumulative Cases for {country} from {start_date} to {end_date}', width=800, height=400)
    fig.show()

cumulative_cases_period('Egypt', '2020-03-01', '2020-05-01')

def cumulative_deaths_period(country, start_date, end_date):
    data = df[(df.country == country) & (df.date_reported >= start_date) & (df.date_reported <= end_date)]
    fig = px.line(data, x='date_reported', y='cumulative_deaths',
             title= f'Cumulative Deaths for {country} from {start_date} to {end_date}', width=800, height=400, markers=True)
    fig.show()

cumulative_deaths_period('Egypt', '2020-03-01', '2020-05-01')

df['month_year'] = df.date_reported.dt.to_period('M').astype(str)

def total_cases(country, start_date = df.date_reported.min(), end_date = df.date_reported.max()):
    data = df[df.country == country]
    fig = px.histogram(data, x='month_year', y='new_cases', title=f'Total Cases for {country}',
         width=800, height=400, range_x=[start_date, end_date], nbins=50)
    fig.show()

total_cases('Egypt')

total_cases('United States of America')

total_cases('Egypt', '2020-03-15', '2021-05-15')

data = df.groupby('country')['new_cases'].sum().sort_values(ascending=False)
px.bar(data, title='Total Cases by Country', width=800, height=800)

data = df.groupby('country')['cumulative_cases'].max().sort_values(ascending=False)
px.bar(data, title='Cumulative Cases by Country', width=800, height=800)

df.groupby('country')['new_cases'].max().nlargest(10)

fig = px.bar(df.groupby('country')['new_cases'].max().nlargest(10), title='Top 10 Countries with the Most New Cases', width=800, height=600)
fig.update(layout_showlegend=False)
fig.update_layout(xaxis_title='Country', yaxis_title='New Cases')
fig.show()

fig = px.bar(df.groupby('country')['new_deaths'].max().nlargest(10),
         title='Top 10 Countries with the Most New Deaths', width=800, height=600, orientation='h')
fig.update(layout_showlegend=False)
fig.update_layout(xaxis_title='Country', yaxis_title='New Deaths')
fig.show()

fig = px.bar(df.groupby('country')['new_deaths'].max().nlargest(10).sort_values(ascending= True),
         title='Top 10 Countries with the Most New Deaths', width=800, height=600, orientation='h')
fig.update(layout_showlegend=False)
fig.update_layout(xaxis_title='Country', yaxis_title='New Deaths')
fig.show()

def top10_countries(data, column):
    data = data.groupby('country')[column].max().nlargest(10)
    fig = px.bar(data, title=f'Top 10 Countries with the Most {column}', width=800, height=600)
    fig.update(layout_showlegend=False)
    fig.update_layout(xaxis_title='Country', yaxis_title=column)
    fig.show()

top10_countries(df, 'new_cases')

top10_countries(df, 'new_deaths')

top10_countries(df, 'cumulative_cases')

top10_countries(df, 'cumulative_deaths')

#FILTER BY REGION

 df.who_region.unique()

'EMRO' in  df.who_region.unique()

df[df.who_region == 'EMRO'].shape

df[df.who_region == 'EMRO'].country.unique()

plt.figure(figsize=(12,6))
sns.lineplot(x='date_reported', y='cumulative_deaths', data=df[df.who_region == 'EMRO'])
plt.title('Total Deathes Cases Over Time')
plt.show()

px.line(df[df.who_region == 'EMRO'], x='date_reported', y='cumulative_deaths')

px.line(df[df.who_region == 'EMRO'], x='date_reported', y='cumulative_deaths',
         title='Cumulative Deaths Over Time', width=800, height=400)

def cumulative_deaths(who_region):
    fig = px.line(df[df.who_region == who_region], x='date_reported', y='cumulative_deaths',
                    title= f'Cumulative Deaths Over Time for {who_region}', width=800, height=400)
    fig.show()

cumulative_deaths('EMRO')

cumulative_deaths('AMRO')

def cumulative_cases(who_region):
    fig = px.line(df[df.who_region == who_region], x='date_reported', y='cumulative_cases',
                     title= f'Cumulative Cases Over Time for {who_region}', width=800, height=400)
    fig.show()

cumulative_cases('EMRO')

cumulative_cases('AMRO')

def cumulative_cases_period(who_region, start_date, end_date):
    data = df[(df.who_region == who_region) & (df.date_reported >= start_date) & (df.date_reported <= end_date)]
    fig = px.line(data, x='date_reported', y='cumulative_cases', title= f'Cumulative Cases for {who_region} from {start_date} to {end_date}', width=800, height=400)
    fig.show()

def cumulative_deaths_period(who_region, start_date, end_date):
    data = df[(df.who_region == who_region) & (df.date_reported >= start_date) & (df.date_reported <= end_date)]
    fig = px.line(data, x='date_reported', y='cumulative_deaths', title= f'Cumulative Deaths for {who_region} from {start_date} to {end_date}', width=800, height=400)
    fig.show()

cumulative_deaths_period('EMRO', '2020-03-01', '2020-05-01')

data = df[df['who_region'] == 'EMRO'].groupby('country')['cumulative_cases'].max().sort_values(ascending=False)
px.bar(data, title='Cumulative Cases by Country in EMRO Region', width=800, height=800)

data = df[df['who_region'] == 'EMRO'].groupby('country')['cumulative_deaths'].max().sort_values(ascending=False)
fig = px.bar(data, title='Cumulative Deaths by Country in EMRO Region', width=800, height=800)
fig.update(layout_showlegend=False)
fig.show()

def total_cases(who_region, start_date = df.date_reported.min(), end_date = df.date_reported.max()):
    data = df[df.who_region == who_region]
    fig = px.histogram(data, x='month_year', y='new_cases', title=f'Total Cases for {who_region}',
         width=800, height=400, range_x=[start_date, end_date], nbins=50)
    fig.show()

total_cases('EMRO', '2020-03-15', '2021-05-15')

def total_cases(who_region, start_date = df.date_reported.min(), end_date = df.date_reported.max()):
    data = df[df.who_region == who_region]
    fig = px.histogram(data, x='month_year', y='new_cases',  title=f'Total Cases for {who_region}',
         width=800, height=400, range_x=[start_date, end_date], nbins=50)
    fig.show()

total_cases('EMRO')

px.bar(df.groupby('who_region')['new_cases'].sum().sort_values(ascending=False),
         title='Total Cases by WHO Region', width=800, height=400)

total_cases('EMRO', '2020-03', '2021-03')

total_cases('EMRO', '2020-03-15', '2021-05-15')

px.bar(df.groupby('who_region')['new_cases'].sum().sort_values(ascending=False),
         title='Total Cases by WHO Region', width=800, height=400)

df['year'] = df.date_reported.dt.year
df['month'] = df.date_reported.dt.month

df.head()

sns.lineplot(x='month', y='new_cases', data=df[df.year == 2020], color='blue')

plt.figure(figsize=(12,6))
sns.lineplot(x='month', y='new_cases', data=df[df.year == 2020], color='red')
sns.lineplot(x='month', y='new_cases', data=df[df.year == 2021], color='blue')
sns.lineplot(x='month', y='new_cases', data=df[df.year == 2022], color='green')
plt.legend(['2020', '2021', '2022'])
plt.title('New Cases Over Time')

plt.figure(figsize=(12,6))
sns.lineplot(x='month', y='new_cases', hue ='year', data=df, palette='Set1')

plt.figure(figsize=(12,6))
sns.lineplot(x='month', y='new_cases', hue ='year', data=df, palette='Set1')

df[df.year==2022].groupby('month')['new_cases'].mean()

df.groupby('year')['new_cases'].sum().astype(float)

plt.figure(figsize=(12,6))
sns.lineplot(x='month', y='new_cases', hue ='year', data=df, palette='Set1', estimator=np.sum)

df['month_year'] = df.date_reported.dt.to_period('M').astype(str)
df.sample(5)

px.bar(df, x='month_year', y='new_cases', title='New Cases Over Time')

px.histogram(df, x='month_year', y='new_cases', color='year', title='New Cases Over Time')

df[df['new_deaths'] < 0]

df[df['new_cases'] < 0]

df[df.new_cases == df.new_cases.max()]

df.query('new_cases == new_cases.max()')

max_new_cases = df[df.new_cases == df.new_cases.max()]
max_new_cases['country']

max_new_cases['country'].iloc[0]

max_new_cases['country'].values

max_new_cases['country'].values[0]

print(f'The country with the most new cases is {max_new_cases["country"].iloc[0]} ')
print(f'The date with the most new cases is {max_new_cases["date_reported"].iloc[0]} ')

max_new_cases.date_reported.dt.day_name()

max_new_cases.date_reported.dt.day_name().values[0]

max_new_cases.date_reported.dt.strftime('%d-%m-%Y')

max_new_cases.date_reported.dt.strftime('%d-%m-%Y').iloc[0]

max_new_cases.date_reported.iloc[0]

df.date_reported = df.date_reported.dt.strftime('%d-%m-%Y')

df.date_reported.dtype

max_new_cases = df[df.new_cases == df.new_cases.max()]
max_new_cases.date_reported.iloc[0]

from plotly.subplots import make_subplots
import plotly.graph_objects as go

data = df.groupby('country').max()
data

fig = make_subplots(rows=2, cols=2, subplot_titles=('New Cases', 'New Deaths', 'Cumulative Cases', 'Cumulative Deaths'))

fig.add_trace(go.Bar(x=data['new_cases'].nlargest(10).index, y=data['new_cases'].nlargest(10).values), row=1, col=1)
fig.add_trace(go.Bar(x=data['new_deaths'].nlargest(10).index, y=data['new_deaths'].nlargest(10).values), row=1, col=2)
fig.add_trace(go.Bar(x=data['cumulative_cases'].nlargest(10).index, y=data['cumulative_cases'].nlargest(10).values), row=2, col=1)
fig.add_trace(go.Bar(x=data['cumulative_deaths'].nlargest(10).index, y=data['cumulative_deaths'].nlargest(10).values), row=2, col=2)

fig.update(layout_showlegend=False)
fig.update_layout(height=900, width=1200, title_text='Top 10 Countries with the Most Cases and Deaths')
fig.show()

df_all_max = df.groupby('country').max()
df_2021_max = df[df.year == 2021].groupby('country').max()
df_2022_max = df[df.year == 2022].groupby('country').sum()

px.choropleth(df_all_max, locations=df_all_max.index, locationmode='country names', color=df_all_max['cumulative_cases'],
            width= 1000, height= 600, title='World wide Cumulative Cases covid19 cases', color_continuous_scale='Reds')  # Try Greens, Purples, Blues

data = dict(type = 'choropleth',
           locations = df_2021_max.index,
            locationmode = 'country names',
            z = df_2021_max['cumulative_cases'],
            text = df_2021_max.index,
            colorscale= 'agsunset',
            reversescale = False,
            marker = dict(line = dict(color='white',width=1)),
            colorbar = {'title':'cumulative_cases'})

layout = dict(title = 'World wide Cumulative Cases in 2021',
             geo = dict(showframe = False, projection = {'type':'mollweide'}))   # equirectangular

go.Figure(data=[data],layout=layout)

def plot_map(data, column):
    data = dict(type = 'choropleth',
           locations = data.index,
            locationmode = 'country names',
            z = data[column],
            text = data.index,
            colorscale= 'agsunset',
            reversescale = False,
            marker = dict(line = dict(color='white',width=1)),
            colorbar = {'title': column } )

    layout = dict(title = f'World wide {column} covid19 cases',
             geo = dict(showframe = False, projection = {'type':'natural earth'}))

    choromap = go.Figure(data=[data],layout=layout)
    choromap.update_layout(margin={"r":0,"t":0,"l":0,"b":0}, width=1000, height=600)
    return(choromap)
plot_map(df_2021_max, 'cumulative_cases')

plot_map(df_all_max, 'cumulative_cases')

plot_map(df_all_max, 'cumulative_deaths')

choromap = px.choropleth(df_2022_max, locations=df_2022_max.index, locationmode='country names', color=df_2022_max['new_cases'],
            width= 1000, height= 600, title='World wide Cumulative Cases covid19 cases', color_continuous_scale='Reds')  # Try Greens, Purples, Blues

choromap.show()

!pip install -U kaleido
Collecting kaleido
 
choromap.write_html('choromap.html')

!pip install dash
import dash
#from dash import html, dcc
import dash_core_components as dcc
import dash_html_components as html

fig = px.choropleth(df_2022_max, locations=df_2022_max.index, locationmode='country names', color=df_2022_max['new_cases'],
            width= 1000, height= 600, title='World wide Cumulative Cases covid19 cases', color_continuous_scale='Reds')  # Try Greens, Purples, Blues
fig2 = plot_map(df_all_max, 'cumulative_cases')

!pip install jupyter-dash
import dash_core_components as dcc
import dash_html_components as html
from jupyter_dash import JupyterDash

fig = px.choropleth(df_2022_max, locations=df_2022_max.index, locationmode='country names', color=df_2022_max['new_cases'],
             width= 1000, height= 600, title='World wide Cumulative Cases covid19 cases', color_continuous_scale='Reds')  # Try Greens, Purples, Blues
fig2 = plot_map(df_all_max, 'cumulative_cases')

# Task 1: Create a new column called 'season' that specify {'Winter', 'Spring', 'Summer', 'Autumn'}

def get_season(month):
    if month in [12, 1, 2]:
        return 'Winter'
    elif month in [3, 4, 5]:
        return 'Spring'
    elif month in [6, 7, 8]:
        return 'Summer'
    else:
        return 'Autumn'

df['season'] = df['month'].apply(get_season)

df.groupby('season')[['new_cases', 'new_deaths']].sum()

data = df.groupby('season')[['new_cases', 'new_deaths']].sum()

px.bar(data, x=data.index, y='new_cases', title=' Total new cases per season')

px.bar(data, x=data.index, y='new_deaths', title=' Total new deaths per season')

df.groupby(['who_region', 'season'])[['new_cases', 'new_deaths']].sum()

for region in df['who_region'].unique():
    data = df[df['who_region'] == region].groupby('season')[['new_cases', 'new_deaths']].sum()
    px.bar(data, x=data.index, y='new_cases', title=' Total new cases per season in EMRO')

def seasons_per_region(region):
    data = df[df['who_region'] == region].groupby('season')[['new_cases', 'new_deaths']].sum()
    # subplots
    fig = make_subplots(rows=1, cols=2, subplot_titles=('Total new cases per season', 'Total new deaths per season'))
    fig.add_trace(go.Bar(x=data.index, y=data['new_cases']), row=1, col=1)
    fig.add_trace(go.Bar(x=data.index, y=data['new_deaths']), row=1, col=2)
    fig.update_layout(title_text=f'Total new cases and deaths per season in {region}')
    fig.show()

for region in df['who_region'].unique():
    seasons_per_region(region)
