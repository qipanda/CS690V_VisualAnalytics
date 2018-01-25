import spacy
import numpy as np
import pandas as pd
from pandas import read_csv
from bokeh.plotting import figure, show
import bokeh.palettes
from bokeh.models.mappers import LinearColorMapper
from bokeh.io import show, output_file
from bokeh.models import Plot, Range1d, FactorRange, MultiLine, Circle, HoverTool, TapTool, BoxSelectTool, BoxZoomTool, PanTool,WheelZoomTool, LabelSet, ColorBar, TapTool, PrintfTickFormatter
from bokeh.models.glyphs import Text
from bokeh.palettes import Spectral4
from bokeh.plotting import figure, curdoc, ColumnDataSource
from bokeh.models.tickers import FixedTicker
from bokeh.models.widgets import Dropdown, MultiSelect, Button, PreText, Select
from math import pi
from bokeh.models import CustomJS, Slider
from bokeh.layouts import row, column, gridplot, layout
import random
from collections import Counter
import operator
from datetime import datetime as dt
import itertools

from bokeh.models import (
    ColumnDataSource,
    HoverTool,
    LinearColorMapper,
    BasicTicker,
    PrintfTickFormatter,
    ColorBar,
)
# import matplotlib as mpl
# import matplotlib.cm as cm
from bokeh.models import SingleIntervalTicker, LinearAxis
# from nltk.sentiment.vader import SentimentIntensityAnalyzer
# import matplotlib as mpl
# import matplotlib.cm as cm
import re
import copy

# import ipdb


def heatmap(cc_data):
	person_list = cc_data.FullName.unique()
	dvals = [i for i in range(0,person_list.shape[0])]
	dkeys = list(person_list)
	person_index = dict(zip(dkeys, dvals))
	cooccur_group = cc_data.groupby(['location','day','hr'])
	cooccur_mat = np.zeros((person_list.shape[0],person_list.shape[0]))
	for name, group in cooccur_group:
		ind = [person_index[i] for i in group.FullName.unique()]
		inc_ind_list = list(itertools.permutations(ind,2))
		for tup in inc_ind_list:
			cooccur_mat[tup]+=1
	return cooccur_mat, person_list



# Debugging
# import ipdb
# ipdb.set_trace()

# Reading data
cc_data = read_csv('VASTChal2014MC2-20140430/cc_data.csv',encoding='iso-8859-1')
poi=read_csv('poi.csv')
# cc_data['FullName']=cc_data[]
# Splitting date_time into date and time
timestamps=pd.DataFrame()
timestamps['ts']=cc_data['timestamp']
datetime_split_df = pd.DataFrame(timestamps.ts.str.split(' ',1).tolist(),columns = ['date','time'])
cc_data['date'] = datetime_split_df['date']
cc_data['time'] = datetime_split_df['time']

# Splitting date into day, month and year
# Storing day as 'day_x' to be plotted in x-axis in the plot
date_split_df = pd.DataFrame(datetime_split_df.date.str.split('/').tolist(),columns = ['month','day','year'])
cc_data['day'] = date_split_df['day'].astype(float)
# Splitting time into hours and minutes and converting then to floating point numbers to be plotted on the y-axis
time_split_df = pd.DataFrame(datetime_split_df.time.str.split(':').tolist(),columns = ['hr','minute'])
time_split_df['time_y'] = time_split_df['hr'].astype(float) + ((time_split_df['minute'].astype(float))*(5/300))
cc_data['time_y'] = time_split_df['time_y']
cc_data['hr'] = time_split_df['hr']
cc_data['FullName'] = cc_data['FirstName'].astype(str) + " " + cc_data['LastName'].astype(str)

# ipdb.set_trace()
cc_data['day_x'] = cc_data['day'] + np.random.uniform(-0.4,0.4,(cc_data['day'].shape[0],))
cc_data['color']='blue'
for i in range(cc_data['LastName'].count()):
	if cc_data.iloc[i]['LastName'] in list(poi['per']):
		cc_data.at[i,'color']='firebrick'

# Grouping data by locations
loc_group = cc_data.groupby(['location'])


# Saving the locations list to be passed to the dropdown menu
location_list = list(loc_group.groups.keys())

cooccur_mat, person_list = heatmap(cc_data)
heat_df = pd.DataFrame(cooccur_mat)
heat_df.columns=person_list
heat_df.index=person_list
heat_df.columns.name='colnames'
heat_df.index.name='rownames'
xLabels = list(heat_df.index)
yLabels = list(heat_df.columns)
heat_plot_df = pd.DataFrame(heat_df.stack(dropna=False),columns=["colocation"]).reset_index()
colors = ["#75968f", "#a5bab7", "#c9d9d3", "#e2e2e2", "#dfccce", "#ddb7b1", "#cc7878", "#933b41", "#550b1d"]
mapper = LinearColorMapper(palette=colors, low=heat_plot_df.colocation.min(), high=heat_plot_df.colocation.max())


heatmap_source = ColumnDataSource(heat_plot_df)
TOOLS = "hover,save,pan,box_zoom,reset,wheel_zoom"

hp = figure(title="Colocation heatmap",
           x_range=xLabels, y_range=yLabels,
           x_axis_location="above",y_axis_location="right", plot_width=800, plot_height=600,
           tools=TOOLS, toolbar_location='below')

hp.grid.grid_line_color = None
hp.axis.axis_line_color = None
hp.axis.major_tick_line_color = None
hp.axis.major_label_text_font_size = "5pt"
hp.axis.major_label_standoff = 0
hp.xaxis.major_label_orientation = pi / 3

hp.rect(x="rownames", y="colnames", width=1, height=1,
       source=heatmap_source,
       fill_color={'field': 'colocation', 'transform': mapper},
       line_color=None)

color_bar = ColorBar(color_mapper=mapper, major_label_text_font_size="5pt",
                     ticker=BasicTicker(desired_num_ticks=len(colors)),
                     formatter=PrintfTickFormatter(format="%d"),
                     label_standoff=6, border_line_color=None, location=(0, 0))
hp.add_layout(color_bar, 'right')

hp.select_one(HoverTool).tooltips = [
     ('People', '@rownames , @colnames'),
     ('cooccurence freq.', '@colocation'),
]





# Getting frequency of people 
people_freq = loc_group.get_group(location_list[0]).groupby(['FullName']).count()
people_freq.sort_values('location',ascending=0,inplace=True)
stop_ind = min(people_freq.shape[0],7)
people_freq = people_freq[0:stop_ind]
people_freq['y'] = np.arange(0,stop_ind)[::-1]
people_freq['full_name']=people_freq.index.tolist()
# Creating the column datasource for the first (default) location
circle_source = ColumnDataSource(loc_group.get_group(location_list[0]))

# Creating the Column data source for the histogram of first location
hist_source = ColumnDataSource(people_freq)

# Creating the bokeh plot
tools='box_zoom, wheel_zoom, pan, reset, xwheel_pan'
p = figure(tools=tools,plot_width=800, plot_height=600, title="Location Patterns for 1/6/2014 - 1/19/2014",x_range=(0,24),y_range=(5,20))
p.circle(x = 'time_y', y = 'day_x', radius=0.08, alpha=0.7, source = circle_source, name='circ', color='color')
p.yaxis.ticker = FixedTicker(ticks = [i for i in range(5,21)])
p.xaxis.ticker = FixedTicker(ticks = [i for i in range(0,25,2)])
p.yaxis[0].formatter = PrintfTickFormatter(format="1/%d/2014")
p.xaxis[0].formatter = PrintfTickFormatter(format="%d:00")
p.ygrid[0].ticker=FixedTicker(ticks=np.arange(5.5,20.5,1))
p.xgrid[0].ticker=FixedTicker(ticks=np.arange(0,25,2))

# Creatin hbar plot object
p2 = figure(tools=tools,plot_width=500, plot_height=300, title="Frequent visitors to this place")
p2.hbar(y = 'y', right = 'location', left=0, height=0.8, color="firebrick", alpha=0.5, source = hist_source, name='hist')


# p2.y_range = FactorRange(factors=list(people_freq['full_name']))
# p.yaxis.major_label_orientation = pi/4
# Update function for the dropdown tool
def update_by_location(attr,old,new):
	location = select_location.value
	new_df = loc_group.get_group(location)
	new_hist_df = loc_group.get_group(location).groupby(['FullName']).count()
	new_hist_df.sort_values('location',ascending=0,inplace=True)
	stop_ind = min(new_hist_df.shape[0],7)
	new_hist_df = new_hist_df[0:stop_ind]
	new_hist_df['y'] = np.arange(0,stop_ind)[::-1]
	new_hist_df['full_name']=new_hist_df.index.tolist()
	new_source = ColumnDataSource(new_df)
	new_hist_source = ColumnDataSource(new_hist_df)
	circle_source.data = new_source.data
	hist_source.data = new_hist_source.data
	# p2.y_range = FactorRange(factors=list(new_hist_df['full_name']))

# Creating dropdown tool
select_location = Select(title='Select Location :', value=location_list[0], options=location_list)
select_location.on_change('value',update_by_location)

# Creating Hover Tool for p
p.add_tools(HoverTool(tooltips = [
	('First Name', '@FirstName')
	,('Last Name', '@LastName')
	,('Expense', '@price')
]
	,renderers =[p.select('circ')[0]]
))

# Creating Hover Tool for p2
p2.add_tools(HoverTool(tooltips = [
	('Full Name', '@full_name')
]
	,renderers =[p2.select('hist')[0]]
))

curdoc().add_root(column(row(p,hp),row(p2,select_location)))
