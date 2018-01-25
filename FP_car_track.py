import pandas as pd
import numpy as np
import shapefile
import pickle
from datetime import datetime

# Bokeh imports
import bokeh.palettes
from bokeh.io import curdoc
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, HoverTool, LabelSet, Arrow, VeeHead, ColorBar, GlyphRenderer, Div
from bokeh.layouts import layout, widgetbox, row, column
from bokeh.models.mappers import CategoricalColorMapper, LinearColorMapper
from bokeh.models.tickers import FixedTicker
from bokeh.models.widgets import Button, Slider, RadioGroup, RadioButtonGroup, DataTable, TableColumn

'''Load the geotracking data, round to nearest Minute to reduce large amount of rows'''
# read the csv, convert first col to datetime format
# df_geotrack = pd.read_csv('VASTChal2014MC2-20140430/gps.csv')
# df_geotrack['Timestamp'] = pd.to_datetime(arg=df_geotrack['Timestamp'])
# df_geotrack.to_pickle('gps.pickle')

# load the timeseries version of the gps data as a df
df_geotrack = pd.read_pickle('gps.pickle')
df_geotrack['Minute'] = df_geotrack['Timestamp'].apply(lambda dt: datetime(dt.year, dt.month, dt.day, dt.hour, dt.minute))

# filter each row down to the first entry per Minute and id
lowest_Minute_filter = df_geotrack.groupby(['id', 'Minute'])['Timestamp'].min().to_frame().reset_index()
df_geotrack = df_geotrack.merge(right=lowest_Minute_filter, how='inner', on=['id', 'Minute', 'Timestamp'])

# some rows have the different lat long but the same timestamp and id, filter those out
df_geotrack['SUM'] = df_geotrack['lat'] + df_geotrack['long']
latlong_filter = df_geotrack.groupby(['id', 'Minute', 'Timestamp'])['SUM'].min().to_frame().reset_index()
df_geotrack = df_geotrack.merge(right=latlong_filter, how='inner', on=['id', 'Minute', 'Timestamp', 'SUM'])
df_geotrack.drop(['SUM'], axis=1, inplace=True)

# join geo data with employee data
df_employees = pd.read_csv('VASTChal2014MC2-20140430/car-assignments.csv')
df_geotrack = df_geotrack.merge(right=df_employees, how='left', left_on='id', right_on='CarID')

# order the Minutes by labelling them with an integer
df_geotrack['Time_Order'] = df_geotrack['Minute'].rank(method='dense').astype(np.int64)

'''Create the timeseries plot of people moving in the lat long'''
# create the datasource for each car into a CDS
source_cars = ColumnDataSource({
    'id':[]
    ,'Minute':[]
    ,'lat':[]
    ,'long':[]
    ,'LastName':[]
    ,'FirstName':[]
    ,'CurrentEmploymentType':[]
    ,'CurrentEmploymentTitle':[]
})

# create the figure
fig = figure(
    title='GASTech Car GPS Tracking Pre Abduction'
    ,plot_width=500
    ,plot_height=500
    ,x_range=(24.82, 24.92)
    ,y_range=(36.04, 36.10)
)
fig.xaxis.axis_label = 'Longitude'
fig.yaxis.axis_label = 'Latitude'

# create the map shapes
# load map data and get the shapes
sf_Kronos_Island = shapefile.Reader("VASTChal2014MC2-20140430/Geospatial/Kronos_Island")
sf_Abila = shapefile.Reader("VASTChal2014MC2-20140430/Geospatial/Abila")

# map of kronos island
kronos_pts = sf_Kronos_Island.shapes()[0].points
kronos_x_pts = [coord[0] for coord in kronos_pts]
kronos_y_pts = [coord[1] for coord in kronos_pts]
fig.patch(x=kronos_x_pts, y=kronos_y_pts, fill_alpha=0.25, fill_color='green', line_color='black')

# all shapes in Abila
abila_shapes = sf_Abila.shapes()
abila_pts = [shape.points for shape in abila_shapes]
abila_x_pts = []
abila_y_pts = []

for pts in abila_pts:
    abila_x_pts.append([coord[0] for coord in pts])
    abila_y_pts.append([coord[1] for coord in pts])

fig.patches(xs=abila_x_pts, ys=abila_y_pts, fill_alpha=0.25, line_alpha=0.25, fill_color='gray', line_color='grey')

# create labels for shapes with > 2 points
xs = []
ys = []
shape_label = []
abila_records = sf_Abila.records()

for shape, record in zip(abila_shapes, abila_records):
    if len(shape.points) > 3:
        xs.append(np.mean([coord[0] for coord in shape.points])) # calc the centroid
        ys.append(np.mean([coord[1] for coord in shape.points]))
        shape_label.append(record[2]) # extract the label, 3rd index is the name

source_shape = ColumnDataSource({
    'label':pd.Series(shape_label)
    ,'x':pd.Series(xs)
    ,'y':pd.Series(ys)
})

shape_labels = LabelSet(x='x', y='y', text='label', source=source_shape, text_font_size='7pt', text_color='purple', text_alpha=0.75)
fig.add_layout(shape_labels)

# create the dots on the map
fig.circle(
    source=source_cars
    ,x='long'
    ,y='lat'
    ,name='cars'
)

# create sliding point for avg dist
slide_source = ColumnDataSource({
    'x':[24.87]
    ,'y':[36.07]
})
fig.square(
    source=slide_source
    ,x='x'
    ,y='y'
    ,color='red'
)

# add moving labels with the cars using the last names of drivers
labels = LabelSet(x='long', y='lat', text='LastName', source=source_cars, text_font_size='9pt')
fig.add_layout(labels)

# add hover
fig.add_tools(HoverTool(tooltips=[
    ('ID', '@id')
    ,('FirstName:', '@FirstName')
    ,('LastName:', '@LastName')
    ,('EmploymentType:', '@CurrentEmploymentType')
    ,('EmploymentTitle', '@CurrentEmploymentTitle')
]
    ,show_arrow=True
    ,renderers=[fig.select('cars')[0]]
))

def update_car_loc(attr, old, new):
    # change the car loc based on the time value
    t_cur = round(time_slider.value)

    # filter to current time
    df_filtered = df_geotrack.loc[df_geotrack['Time_Order'] <= t_cur, :]
    min_tm_filter = df_filtered.groupby('id')['Time_Order'].max().to_frame().reset_index()
    df_filtered = df_filtered.merge(right=min_tm_filter, how='inner', on=['id', 'Time_Order'])

    # update the displayed datetime
    cur_dt = df_filtered['Timestamp'].iloc[0]
    time_slider_datetime.text = """
    <body>
        Current Datetime: {}/{}/{} | Hour: {}, Minute: {}
    </body>
    """.format(cur_dt.year, cur_dt.month, cur_dt.day, cur_dt.hour, cur_dt.minute)

    # update the car cds
    source_cars.data = {
        'id':df_filtered['id']
        ,'Minute':df_filtered['Minute']
        ,'lat':df_filtered['lat']
        ,'long':df_filtered['long']
        ,'LastName':df_filtered['LastName']
        ,'FirstName':df_filtered['FirstName']
        ,'CurrentEmploymentType':df_filtered['CurrentEmploymentType']
        ,'CurrentEmploymentTitle':df_filtered['CurrentEmploymentTitle']
    }

    # update top cum average distances to selected point
    df_dist = df_filtered.groupby(['id', 'FirstName', 'LastName'])['lat', 'long'].mean().reset_index()
    df_dist['cumulative_mean_dist'] = ((df_dist['long'] - x_slider.value)**2.0 + (df_dist['lat'] - y_slider.value)**2.0)**0.5
    slide_source.data = {
        'x':[x_slider.value]
        ,'y':[y_slider.value]
    }

    # display top closest to reference by sorting first
    df_dist.sort_values(by=['cumulative_mean_dist'], ascending=True, inplace=True)
    table_source.data = {
        'FirstName':df_dist['FirstName'][0:10]
        ,'LastName':df_dist['LastName'][0:10]
        ,'Mean_Cumulative_Distance_From_Reference':df_dist['cumulative_mean_dist'][0:10]
    }

def play_update():
    # If the slider has not reached the end, increment and recalc graph
    if time_slider.value != time_slider.end:
        time_slider.value += 1 # will auto update with callback
    else:
        # if the slider has reached the end we want to stop (remove the periodic_callback)
        animate()

def animate():
    # allow for playing and pausing duing the animations and pre animation
    if play_button.label == '► Play':
        play_button.label = '❚❚ Pause'
        curdoc().add_periodic_callback(play_update, play_speed_vals[play_speed_radiogroup.active])
    else:
        play_button.label = '► Play'
        curdoc().remove_periodic_callback(play_update)

time_slider = Slider(start=1, end=df_geotrack['Time_Order'].max(), value=1, step=1, title='Minute:', width=500)
time_slider.on_change('value', update_car_loc)

slider_title_div = Div(text="""<b>Mean Distance Reference:</b>""")
x_slider = Slider(start=24.82, end=24.92, value=24.87, step=0.001, title='long:', width=500)
y_slider = Slider(start=36.04, end=36.10, value=36.07, step=0.001, title='lat:', width=500)
x_slider.on_change('value', update_car_loc)
y_slider.on_change('value', update_car_loc)

# display the top 10 closest
table_div = Div(text="""<b>Closest Cars on Average (Cumulativly Over Time) to Selected Point""")
table_source = ColumnDataSource({
    'FirstName':[]
    ,'LastName':[]
    ,'Mean_Cumulative_Distance_From_Reference':[]
})
columns = [
    TableColumn(field='FirstName', title='FirstName')
    ,TableColumn(field='LastName', title='LastName')
    ,TableColumn(field='Mean_Cumulative_Distance_From_Reference', title='Mean_Cumulative_Distance_From_Reference')
]
slider_table = DataTable(source=table_source, columns=columns, width=500, height=300)

time_slider_datetime = Div()

play_speeds_radio_title = Div(text="""<b>Play Speed:</b>""")
play_speed_vals = [250, 125, 50]
play_speeds = [
    'Slow ({}ms delay)'.format(play_speed_vals[0])
    , 'Medium ({}ms delay)'.format(play_speed_vals[1])
    , 'Fast ({}ms delay)'.format(play_speed_vals[2])]
play_speed_radiogroup = RadioButtonGroup(labels=play_speeds, active=1, width=500)

# play/pause button
play_button = Button(label='► Play', width=500)
play_button.on_click(animate)

righthand_widgets = widgetbox(children=[
    play_speeds_radio_title, play_speed_radiogroup, time_slider, time_slider_datetime, play_button, slider_title_div, x_slider, y_slider, table_div, slider_table])

# create the layout for sliders and plot
layout_all = layout(sizing_mode='scale_width', children=[
    [fig, righthand_widgets]
])

# update once before running
update_car_loc('value', 0, 0)

# for bokeh server to run
doc = curdoc()
doc.add_root(layout_all)
