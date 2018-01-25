import pandas as pd
import numpy as np
import time
import matplotlib as mpl
import matplotlib.cm as cm

# NLP tools and NaiveBayesClassifier for prediction pos/neg sentiments
import nltk
from nltk.classify import NaiveBayesClassifier

# Bokeh imports
import bokeh.palettes
from bokeh.io import curdoc
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, HoverTool, LabelSet, Arrow, VeeHead, ColorBar, GlyphRenderer, Div
from bokeh.layouts import layout, widgetbox, row, column
from bokeh.models.mappers import CategoricalColorMapper, LinearColorMapper
from bokeh.models.tickers import FixedTicker
from bokeh.models.widgets import Button, Slider, RadioGroup, RadioButtonGroup

'''
Model code based on tutorial at:
https://www.twilio.com/blog/2017/09/sentiment-analysis-python-messy-data-nltk.html
Data from:
https://www.kaggle.com/c/si650winter11/data
and
@InProceedings{Pang+Lee:05a,
  author =       {Bo Pang and Lillian Lee},
  title =        {Seeing stars: Exploiting class relationships for sentiment
                  categorization with respect to rating scales},
  booktitle =    {Proceedings of the ACL},
  year =         2005
}
'''

def format_sentence(sent):
    # word_tokennize splits the sentence (tweet) into words, the binary array allows for indv. word classifications of pos/neg
    return({word: True for word in nltk.word_tokenize(sent)})

'''
1.) Load the twitter data and movie review data to train the NBC model

The NaiveBayesClassifier makes the assumption that features (words) in each tweet have indepedent contribution to the sentiment ie:
    P(word_1,...,word_n | class) = P(word_1|class)*...*P(word_n|class))

Labels are 1 = positive sentiment, 0 = negative sentiment
'''
# load the twitter sentiment data and format the tweets, 1=pos 0=neg
df_twitter = pd.read_csv('mich_twit_sent_data.txt', sep='\t', header=None, names=['label', 'text'])
# load the movie review sentiment
df_movie_pos = df_movie_pos = pd.read_csv('rt-polarity.pos', header=None, sep='\n', engine='python', names=['text'])
df_movie_neg = df_movie_neg = pd.read_csv('rt-polarity.neg', header=None, sep='\n', engine='python', names=['text'])
df_movie_pos['label'] = 1
df_movie_neg['label'] = 0

df = pd.concat([df_twitter, df_movie_pos, df_movie_neg])
df.loc[:, 'text'] = df.loc[:, 'text'].apply(func=format_sentence) # format each tweet for nltk nbc

# train the nbc on full twitter data set
train_data = [[x, y] for x, y in zip(df.loc[:, 'text'].tolist(), df.loc[:, 'label'].tolist())]
nbc = NaiveBayesClassifier.train(train_data)

'''
2.) Load the email dataset to assign integer time labels and classify the headers as pos or neg

df_email has columns:
    - From
    - To
    - Date (datetime object)
    - Time_Order (integer with low being early and high being late, determined by radio options later)
    - Subject
    - Sent
'''
df_email = pd.read_csv('MC1 Data/MC1 Data/email headers.csv', engine='python')

# classify each subject header
df_email['Sent'] = df_email.apply(
    lambda row: 'pos' if nbc.classify(format_sentence(row['Subject'])) else 'neg', axis=1)
df_email['Date'] = pd.to_datetime(df_email['Date']) # convert to datetime

# explode the To into multiple rows
df_email['To'] = df_email['To'].apply(lambda row: row.split(', '))
df_email_exp = df_email['To'].apply(
    lambda x: pd.Series(x)).stack().reset_index(level=1, drop=True).to_frame(name='To')
df_email = pd.merge(
    left=df_email.drop(axis=1, columns='To'), right=df_email_exp, left_index=True, right_index=True).reset_index(drop=True)

'''
3.) Setup dataframes needed for bokeh CDS

need stationary source of:
    - Email (person)
    - (x, y) of circle centered at (0, 0)

    cos(angle) = x/radius -> x = radius*cos(angle)
    sin(angle) = y/radius -> y = radius*sin(angle)

arrow source of:
    - Time order (integer)
    - From Email
    - To Email
    - Subject
    - Sentiment

Join these together per time division chosen (per email, per hour, per day)
'''

# calculate the circle spread and add labels
unique_emails = df_email['From'].append(df_email['To']).unique().tolist()
unique_emails.sort() # to match df_netsent_subset later
radial_spread = (2.0*np.pi)/len(unique_emails)
circle_radius = 5.0
xs = circle_radius*np.cos(radial_spread*np.arange(len(unique_emails)))
ys = circle_radius*np.sin(radial_spread*np.arange(len(unique_emails)))

# create the stationary df and join to df_email
df_stationary = pd.DataFrame({'Emails':unique_emails, 'x':xs, 'y':ys})
df_email = df_email.merge(
    right=df_stationary.rename(columns={'x':'x_from', 'y':'y_from'}), left_on='From', right_on='Emails')
df_email = df_email.merge(
    right=df_stationary.rename(columns={'x':'x_to', 'y':'y_to'}), left_on='To', right_on='Emails')
df_email.drop(axis=1, columns=['Emails_x', 'Emails_y'], inplace=True) # get rid of useless rows

''' 4.) Setup Bokeh figures'''
source_stationary = ColumnDataSource({
    'Emails':df_stationary['Emails']
    ,'x':df_stationary['x']
    ,'y':df_stationary['y']
    ,'Net_Sent':[0]*df_stationary.shape[0]
    ,'Net_Pos':[0]*df_stationary.shape[0]
    ,'Cric_Size':[0]*df_stationary.shape[0]
})
source_arrows = ColumnDataSource({
    'x_from':[]
    ,'y_from':[]
    ,'x_to':[]
    ,'y_to':[]
    ,'xs':[]
    ,'ys':[]
    ,'From':[]
    ,'To':[]
    ,'Subject':[]
    ,'Sentiment':[]
}) # time is not needed as each of these will be an instance of a time to be updated

# create the figure
p = figure(
    title='GASTech Internal Emails Over Time'
    ,plot_width=500
    ,plot_height=500
    ,x_range=(-7, 7)
    ,y_range=(-7, 7)
)

# create the stationary points representing emails (people) and hover for stationary
# create the color mapper for circle pos/neg sent
colormap = cm.get_cmap("RdYlGn")
bokehpalette = [mpl.colors.rgb2hex(m) for m in colormap(np.arange(colormap.N))] # get in terms of hexideci
mapper = LinearColorMapper(palette=bokehpalette, low=-1.0, high=1.0)
p.circle(
    source=source_stationary
    ,x='x'
    ,y='y'
    ,size='Cric_Size'
    ,fill_color={'field':'Net_Sent', 'transform':mapper}
    ,line_color='black'
    ,name='stationary'
)
p.add_tools(HoverTool(tooltips=[
    ('Employee Email:', '@Emails')
]
    ,show_arrow=True
    ,renderers=[p.select('stationary')[0]]
))

# create a reference circle
p.circle(x=[0], y=[0], radius=circle_radius, fill_alpha=0.0, line_alpha=1.0, color='black')

# create the arrows and hovertool layout ontop of multiline
cat10_palette = bokeh.palettes.Category10[10]
arrow_line_palette = [cat10_palette[3], cat10_palette[2]] # red and green
arrow_mapper = CategoricalColorMapper(palette=arrow_line_palette, factors=['neg', 'pos'])
p.add_layout(
    Arrow(
        end=VeeHead(
            size=10
            ,fill_alpha=0.50
            ,line_alpha=0.50
            ,fill_color=cat10_palette[7]
        )
        ,source=source_arrows
        ,x_start='x_from'
        ,y_start='y_from'
        ,x_end='x_to'
        ,y_end='y_to'
        ,line_alpha=0.50
        ,line_color={'field':'Sentiment', 'transform':arrow_mapper}
    )
)
p.multi_line(
    source=source_arrows
    ,xs='xs'
    ,ys='ys'
    ,name='arrow_tags'
    ,line_alpha=0 # make invis, only for hover
)
p.add_tools(HoverTool(tooltips=[
    ('From:', '@From')
    ,('To', '@To')
    ,('Subject:', '@Subject')
    ,('Sentiment', '@Sentiment')
]
    ,show_arrow=True
    ,renderers=[p.select('arrow_tags')[0]]
))

# initialize a dataframe to be used gloablly in updating the stationary points
df_netsent_global = pd.DataFrame({'Time_Order':[], 'Emails':[], 'neg_Sent_Cnt':[], 'pos_Sent_Cnt':[], 'neg_Recv_Cnt':[], 'pos_Recv_Cnt':[], 'Net_Sent':[], 'Net_Pos':[]})

def update_arrows(attr, old, new):
    # get the current time order and select slice at t_cur
    t_cur = round(time_slider.value) # new contains the current slider value

    # update the stationary points
    df_arrows = update_stationary(t_cur)

    # # update the CDS for arrows in a smooth manner TODO fix timing on pause button with anim
    # cur_speed = play_speed_vals[play_speed_radiogroup.active]
    anim_granularity = 50
    anim_fnc = np.arange(anim_granularity)**2
    anim_fnc = anim_fnc/anim_fnc.max()
    for i in anim_fnc:
        source_arrows.data = {
            'x_from':df_arrows['x_from']
            ,'y_from':df_arrows['y_from']
            ,'x_to':pd.Series(df_arrows['x_from'] + i*(df_arrows['x_to'].values - df_arrows['x_from'].values))
            ,'y_to':pd.Series(df_arrows['y_from'] + i*(df_arrows['y_to'].values - df_arrows['y_from'].values))
            ,'xs':pd.Series(df_arrows.loc[:, ['x_from', 'x_to']].values.tolist())
            ,'ys':pd.Series(df_arrows.loc[:, ['y_from', 'y_to']].values.tolist())
            ,'From':df_arrows['From']
            ,'To':df_arrows['To']
            ,'Subject':df_arrows['Subject']
            ,'Sentiment':df_arrows['Sent']
        }

def update_stationary(t_cur):
    # import ipdb; ipdb.set_trace()
    # for calculations of shrinking/color scale
    max_net_sent = df_netsent_global['Net_Sent'].abs().max()
    max_net_pos = df_netsent_global['Net_Pos'].abs().max()
    df_netsent_subset = df_netsent_global.loc[df_netsent_global['Time_Order']==t_cur] # cur time/person net sent/pos

    # update stationary current positions
    df_stationary['x_cur'] =\
        df_stationary['x'] - (df_netsent_subset['Net_Sent'].reset_index(drop=True)/max_net_sent)*df_stationary['x']
    df_stationary['y_cur'] =\
        df_stationary['y'] - (df_netsent_subset['Net_Sent'].reset_index(drop=True)/max_net_sent)*df_stationary['y']

    # update stationary source
    source_stationary.data = {
        'Emails':df_netsent_subset['Emails']
        ,'x':df_stationary['x_cur']
        ,'y':df_stationary['y_cur']
        ,'Net_Sent':df_netsent_subset['Net_Sent']/max_net_sent
        ,'Net_Pos':df_netsent_subset['Net_Pos']
        ,'Cric_Size':pd.Series(18*(df_netsent_subset['Net_Sent'].values/max_net_sent) + 9)
    }

    # only care about df_email in cur time for arrows to display
    df_arrows = df_email.loc[df_email['Time_Order']==t_cur, ['From', 'To', 'Subject', 'Sent']]

    # update email source with new stationary positions for the arrows
    df_arrows = df_arrows.merge(
        right=df_stationary.rename(columns={'x_cur':'x_from', 'y_cur':'y_from'}), left_on='From', right_on='Emails')
    df_arrows = df_arrows.merge(
        right=df_stationary.rename(columns={'x_cur':'x_to', 'y_cur':'y_to'}), left_on='To', right_on='Emails')
    df_arrows.drop(axis=1, columns=['Emails_x', 'Emails_y'], inplace=True) # get rid of useless rows

    return df_arrows

def change_time_gran(*args):
    if time_gran_radio.labels[time_gran_radio.active] == 'Per Email':
        df_email['Time_Order'] = df_email['Date'].rank(method='dense').astype(np.int64)
    elif time_gran_radio.labels[time_gran_radio.active] == 'Per Hour':
        # find the cumulative hours (all days within same month)
        df_email['Time_Order'] = df_email.apply(lambda row: row['Date'].day*24 + row['Date'].hour, axis=1)
        df_email['Time_Order'] = df_email['Time_Order'].rank(method='dense').astype(np.int64)
    elif time_gran_radio.labels[time_gran_radio.active] == 'Per Day':
        # round to the floored day
        df_email['Time_Order'] = df_email.apply(lambda row: row['Date'].day, axis=1)
        df_email['Time_Order'] = df_email['Time_Order'].rank(method='dense').astype(np.int64)

    # update slider max val and title reset to 1
    time_slider.end = df_email['Time_Order'].max()
    time_slider.title = slider_titles[time_gran_radio.active]
    time_slider.value = 1

    '''
    Add to net count of sent/recieved for distance and size aggregation and add to net count of pos/neg for circle color aggregation
        Schema should be: Time_Order, Email, Net Sent, Net Pos
    '''
    # count the number of sent pos/neg emails. recieved pos/neg emails,
    df_sent = df_email.groupby(['Time_Order', 'From', 'Sent'])['Subject'].count().rename('Sent_Cnt').to_frame().unstack(level='Sent', fill_value=0)
    df_recv = df_email.groupby(['Time_Order', 'To', 'Sent'])['Subject'].count().rename('Recv_Cnt').to_frame().unstack(level='Sent', fill_value=0)

    # remove the multi index and rename them, then reset the index to bring index into columns
    df_sent.columns = df_sent.columns.droplevel()
    df_recv.columns = df_recv.columns.droplevel()
    df_sent.reset_index(inplace=True)
    df_recv.reset_index(inplace=True)
    df_sent.rename(columns={'neg':'neg_Sent_Cnt', 'pos':'pos_Sent_Cnt'}, inplace=True)
    df_recv.rename(columns={'neg':'neg_Recv_Cnt', 'pos':'pos_Recv_Cnt'}, inplace=True)

    # outer join to get full neg/pos sent and received emails per time/person combination
    df_netsent = pd.merge(left=df_sent, right=df_recv, how='outer', left_on=['Time_Order', 'From'], right_on=['Time_Order', 'To']).fillna(value=0)

    # convert From/To column into just the name of the non-zero column in df_netsent
    df_netsent.loc[(df_netsent['From']==0), 'From'] = df_netsent.loc[(df_netsent['From']==0), 'To']
    df_netsent.drop(labels='To', axis=1, inplace=True)
    df_netsent.rename(columns={'From':'Emails'}, inplace=True)
    df_netsent = df_netsent.set_index(keys=['Time_Order', 'Emails']).unstack('Emails').fillna(0).cumsum().stack('Emails')
    df_netsent.reset_index(inplace=True)
    df_netsent['Net_Sent'] = \
        (df_netsent['pos_Sent_Cnt'] + df_netsent['neg_Sent_Cnt']) - (df_netsent['pos_Recv_Cnt'] + df_netsent['neg_Recv_Cnt'])
    df_netsent['Net_Pos'] = \
        (df_netsent['pos_Sent_Cnt'] + df_netsent['pos_Recv_Cnt']) - (df_netsent['neg_Sent_Cnt'] + df_netsent['neg_Recv_Cnt'])

    # set to global dataframe
    df_netsent_global.loc[:, :] = df_netsent.loc[:, :]

    # reflect update in the arrows
    update_arrows('value', round(time_slider.value), round(time_slider.value))

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

# add widgets
time_gran_radio_title = Div(text="""<b>Time Granularity:</b>""")
time_gran_radio = RadioGroup(labels=['Per Email', 'Per Hour', 'Per Day'], active=0, width=100)# MUST START AT 0 FOR GLBL
time_gran_radio.on_click(change_time_gran)

slider_titles = ['Chronological Email Number:', 'Hour:', 'Day:']
time_slider = Slider(start=1, end=20, value=1, step=1, title=slider_titles[time_gran_radio.active], width=500)
time_slider.on_change('value', update_arrows)

play_speeds_radio_title = Div(text="""<b>Play Speed:</b>""")
play_speed_vals = [750, 500, 250]
play_speeds = [
    'Slow ({}ms delay)'.format(play_speed_vals[0])
    , 'Medium ({}ms delay)'.format(play_speed_vals[1])
    , 'Fast ({}ms delay)'.format(play_speed_vals[2])]
play_speed_radiogroup = RadioButtonGroup(labels=play_speeds, active=1, width=500)

# play/pause button
play_button = Button(label='► Play', width=500)
play_button.on_click(animate)

righthand_widgets = widgetbox(children=[
    time_gran_radio_title, time_gran_radio, play_speeds_radio_title, play_speed_radiogroup, time_slider, play_button])

# create the layout for sliders and plot
layout_all = layout(sizing_mode='scale_width', children=[
    [p, righthand_widgets]
])

change_time_gran()

# for bokeh server to run
doc = curdoc()
doc.add_root(layout_all)
