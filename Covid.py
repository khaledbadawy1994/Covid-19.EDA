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
