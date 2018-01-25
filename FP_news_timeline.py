import spacy
import numpy as np
import pandas as pd
from pandas import read_csv
from bokeh.plotting import figure, show
import bokeh.palettes
from bokeh.models.mappers import LinearColorMapper
from bokeh.io import show, output_file
from bokeh.models import Plot, Range1d, MultiLine, Circle, HoverTool, TapTool, BoxSelectTool, BoxZoomTool, PanTool,WheelZoomTool, LabelSet, ColorBar, TapTool
from bokeh.models.glyphs import Text
from bokeh.palettes import Spectral4
from bokeh.plotting import figure, curdoc, ColumnDataSource
from bokeh.models.tickers import FixedTicker
from bokeh.models.widgets import Dropdown, MultiSelect, Button, PreText
from math import pi
from bokeh.models import CustomJS, Slider
from bokeh.layouts import row, column, gridplot, layout
import random
from collections import Counter
import operator
import matplotlib as mpl
import matplotlib.cm as cm
from bokeh.models import SingleIntervalTicker, LinearAxis
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import matplotlib as mpl
import matplotlib.cm as cm
import re
import copy


# import ipdb

def make_df(entity_bag_2021, average_sentiment_array, entity_key, direction, spacing):
	words=[]
	fontsize=[]
	y=[]
	x=[]
	sentiment=[]
	hover_name = []
	art_iter = 0
	circ_rad=[]
	for key in entity_bag_2021.keys():
		art_words = entity_bag_2021[key][entity_key]	
		art_words_full = copy.deepcopy(art_words)
		for ii in range(0,len(art_words)):
			max_letters=(12 - ii*1)
			if len(art_words[ii])>max_letters:
				art_words[ii]=art_words[ii][:max_letters-2]+'..'
		# print(entity_bag_2021[str(art)]['PERSON'])
		art_fontsize = np.array([(8 + 2*i) for i in range(1,len(art_words)+1)])
		art_circ_rad = (art_fontsize-1)/2
		fontheight = art_fontsize*0.15
		art_y = [fontheight[i]/2 + fontheight[i+1]/2 for i in range(0,len(art_fontsize)-1)]
		art_y = [0] + art_y
		art_y = np.array(art_y)
		art_y = np.cumsum(art_y)*8
		art_y = list(art_y+80)
		art_fontsize = [str(ii)+"pt" for ii in art_fontsize]
		art_x = [(100 + spacing*art_iter) for i in range(0,len(art_words))]
		art_sentiment = [average_sentiment_array[art_iter] for i in range(0,len(art_words))]
		words = words + art_words
		fontsize = fontsize + art_fontsize
		y = y + art_y
		x = x + art_x
		hover_name = hover_name + art_words_full
		sentiment = sentiment + art_sentiment
		circ_rad = circ_rad + list(art_circ_rad)
		art_iter = art_iter + 1

	worddf = pd.DataFrame()
	worddf['words']=words
	worddf['fontsize']=fontsize
	worddf['circ_rad']=circ_rad
	worddf['x']=x
	worddf['y']=list(np.array(y)*direction)
	worddf['sentiment']=sentiment
	worddf['full_text']=hover_name
	return worddf

def calc_avg_sentiment(page_text, nlp, sent_ana):
	'''Given a row from a dataframe as a pd.Series, calculate the statements average sentiment using spacy pipline and nltk VADER model'''
	doc = nlp(page_text)

	words = [word.orth_ for word in doc if (word.is_stop==False and (word.pos_ not in ['PUNCT', 'SPACE', 'X']))]
	wordfreq = Counter(words)
	freq_words = wordfreq.most_common(2)
	words = [x[0] for x in freq_words]
	# words = pd.Series(words)
	# words = words.count()

	total_sentiment = 0
	num_sentences = 0

	for sentence in doc.sents:
	    total_sentiment += sent_ana.polarity_scores(sentence.orth_)['compound'] #.orth_ gives string representation from spacy Span
	    num_sentences += 1

	return total_sentiment/num_sentences, words

# parsing dates and filtering the articles for 20th and 21st January 2014
def parsedate(d):
	flag = 0;
	words = d.split(' ')
	if len(words)>1:
		if words[0]=='20' and words[1]=='January' and words[2]=='2014':
			flag=1
		elif words[0]=='21'and words[1]=='January' and words[2]=='2014':
			flag=2
		else:
			flag=0
	else:
		if words[0]=='2014/01/20':
			flag=1
		elif words[0]=='2014/01/21':
			flag=2
		else:
			flag=0
	return flag

def make_entity_bag(article_list,nlp,sent_ana,poi):
	average_sentiment_array = []
	entity_bag = {}
	for i in article_list:
		entity_bag[str(i)]={}
		entity_bag[str(i)]['HISTORIC']=[]
		entity_bag[str(i)]['PERSON']=[]
		entity_bag[str(i)]['ORG']=[]
		entity_bag[str(i)]['LOC']=[]
		entity_bag[str(i)]['GPE']=[]
		entity_bag[str(i)]['DATE']=[]
		entity_bag[str(i)]['EVENT']=[]
		entity_bag[str(i)]['TIME']=[]
		entity_bag[str(i)]['MONEY']=[]
		entity_bag[str(i)]['FAC']=[]
		entity_bag[str(i)]['WORK_OF_ART']=[]
		entity_bag[str(i)]['NORP']=[]
		print("Reading article " + str(i),end='\r')
		file = open("MC1 Data/MC1 Data/articles/"+str(i) + ".txt","r",encoding='ISO-8859-1')
		text = file.read()
		avg_sent, words = calc_avg_sentiment(text, nlp, sent_ana)
		average_sentiment_array.append(avg_sent)
		doc=nlp(text)
		for ent in doc.ents:
			if ent.label_ in entity_bag[str(i)].keys():
				entity_bag[str(i)][ent.label_].append(ent.text)
		# ipdb.set_trace()
		for word in doc:
			if (word.is_stop==False) and (str(word).strip() in list(poi['per'])):
				entity_bag[str(i)]['HISTORIC'].append(str(word).strip())
		for key in entity_bag[str(i)].keys():
			local_word_list = entity_bag[str(i)][key]
			freq_dict = dict(Counter(local_word_list))
			sorted_freq_dict = sorted(freq_dict.items(), key=operator.itemgetter(1))
			sorted_words = [x[0] for x in sorted_freq_dict]
			entity_bag[str(i)][key]=sorted_words
			if not entity_bag[str(i)][key]:
				entity_bag[str(i)][key]=["None"]
	return entity_bag, average_sentiment_array


def sort_by_date(datedf):
	dtlist = datedf['dates'].tolist()
	year = []
	month = []
	day = []
	month_dict = {
		'January' : 1,
		'February' : 2,
		'March' : 3,
		'April' : 4,
		'May' : 5,
		'June' : 6,
		'July' : 7,
		'August' : 8,
		'September' : 9,
		'October' : 10,
		'November' : 11,
		'December' : 12
	}
	for date in dtlist:
		words = date.split(' ')
		if len(words)==3 and len(words[0])<=2 and not any(c.isalpha() for c in words[0]) and len(words[1])>=3 and len(words[2])==4 and not any(c.isalpha() for c in words[2]):
			year.append(int(words[2]))
			month.append(month_dict[words[1]])
			day.append(int(words[0]))
		else:
			date = date.strip()
			words = date.split('/')
			if len(words)==3 and len(words[0])==4 and len(words[1])<=2 and len(words[2])<=2 and not any(c.isalpha() for c in date):
				year.append(int(words[0]))
				month.append(int(words[1]))
				day.append(int(words[2]))
			else:
				year.append(2015)
				month.append(13)
				day.append(32)
	datedf['year'] = year
	datedf['month'] = month
	datedf['day'] = day
	datedf = datedf.sort_values(['year','month','day'],ascending=[True,True,True])
	return datedf['articles'].tolist(), datedf['dates'].tolist()

# load spacy pipeline and data
nlp = spacy.load('en') 
sent_ana = SentimentIntensityAnalyzer()
article_id = list()
entities = list()
num_articles = 845
top_words_list = []
poi = read_csv('poi.csv')
nlp.vocab[" "].is_stop = True
nlp.vocab["."].is_stop = True
nlp.vocab[","].is_stop = True
nlp.vocab[", "].is_stop = True
nlp.vocab[" ,"].is_stop = True
nlp.vocab[". "].is_stop = True
nlp.vocab[" ."].is_stop = True
# creating article arrays for articles belonging to the two days vs older articles
article_20 = []
article_21 = []
article_2021 = []
article_hist = []
article_hist_dates = []
print("\nFiltering articles dated 20th and 21st January 2014...")
for i in range(0,num_articles):
	local_word_list = []
	print("reading article " + str(i),end='\r')
	file = open("MC1 Data/MC1 Data/articles/"+str(i) + ".txt","r",encoding='ISO-8859-1')
	text = file.read()
	# text = bytes(text, 'utf-8').decode('utf-8', 'ignore')
	doc=nlp(text)
	date = ''
	for ent in doc.ents:
		if ent.label_=='DATE':
			date = ent.text
			break
	if parsedate(date)==1:
		article_20.append(i)
	elif parsedate(date)==2:
		article_21.append(i)
	else:
		article_hist.append(i)
		article_hist_dates.append(date)
print("\n")
datedf = pd.DataFrame()
datedf['articles']=article_hist
datedf['dates']=article_hist_dates

article_hist, date_list = sort_by_date(datedf)

article_2021 = article_20 + article_21

breakpoint_2021 = len(article_20)

print("No. of articles dated 20th and 21st January 2014 : ",len(article_2021))
print("No. of articles dated before                     : ",len(article_hist))

ent_name={}
ent_name['HISTORIC']='Historically important'
ent_name['PERSON']='Person'
ent_name['ORG']='Organization'
ent_name['LOC']='Location'
ent_name['GPE']='Geo-Political'
ent_name['DATE']='Date'
ent_name['EVENT']='Event'
ent_name['TIME']='Time'
ent_name['MONEY']='Currency'
ent_name['FAC']='Facility'

ent_key={}
ent_key['Historically important']='HISTORIC'
ent_key['Person']='PERSON'
ent_key['Organization']='ORG'
ent_key['Location']='LOC'
ent_key['Geo-Political']='GPE'
ent_key['Date']='DATE'
ent_key['Event']='EVENT'
ent_key['Time']='TIME'
ent_key['Currency']='MONEY'
ent_key['Facility']='FAC'
# create bag of words and rank words according to the frequency for each article from 20th and 21st jan 2014
print("\nRecognizing and tagging entities for articles dated 20th and 21st January 2014....")
print("Entity classes : ",'Historically important','Person,','Organization,','Location,','Geo-Political,','Date,','Event,','Time,','Currency,','Facility','\n')

# making the ColumnDataSource
entity_bag_2021, average_sentiment_array = make_entity_bag(article_2021,nlp,sent_ana,poi)
entity_bag_hist, average_sentiment_array_hist = make_entity_bag(article_hist,nlp,sent_ana,poi)

spacing = 120
worddf_up = make_df(entity_bag_2021,average_sentiment_array,'PERSON',1,spacing)

wordsource_up = ColumnDataSource(worddf_up)

worddf_down = make_df(entity_bag_2021,average_sentiment_array,'ORG',-1,spacing)

wordsource_up = ColumnDataSource(worddf_up)
wordsource_down = ColumnDataSource(worddf_down)

linesource = ColumnDataSource({
		'x' : pd.Series([(100 + spacing*i) for i in range(0,len(article_2021))])
		,'sentiment' : pd.Series(average_sentiment_array)
	})

circlesource = ColumnDataSource({
		'x' : pd.Series([(100 + spacing*i) for i in range(0,len(article_2021))])
		,'sentiment' : pd.Series(average_sentiment_array)
	})

xs = [[(100 + spacing*i),(100 + spacing*i)] for i in range(0,len(article_2021))]
ys = [[0,60] for i in range(0,len(article_2021))]

multilinesource1 = ColumnDataSource({
		'xs' : xs,
		'ys' : ys,
		'sentiment' : average_sentiment_array
	})
multilinesource2 = ColumnDataSource({
		'xs' : xs,
		'ys' : [[0,-1*60] for i in range(0,len(article_2021))],
		'sentiment' : average_sentiment_array
	})

colormap = cm.get_cmap("RdYlGn")
bokehpalette = [mpl.colors.rgb2hex(m) for m in colormap(np.arange(colormap.N))]
mapper = LinearColorMapper(palette=bokehpalette, low=-1.0, high=1.0)


tools='tap, wheel_zoom, pan, reset, xwheel_pan'
p = figure(tools=tools,plot_width=1000, plot_height=800, title="News articles clustering based on bag of words overlap",y_range=(-400,400),x_range=(0,800))

p.line(x='x', y=0, line_width=2, source=linesource)
p.circle(x='x',y=0,radius=5,color={'field':'sentiment', 'transform':mapper},source=circlesource, name='cr')
p.circle(x = 'x',y = 'y',color='black', radius='circ_rad', source=wordsource_up, name='circ_up')
p.circle(x = 'x',y = 'y',color='black', radius='circ_rad', source=wordsource_down, name='circ_down')
p.multi_line(xs='xs',ys='ys',line_color={'field' : 'sentiment', 'transform' : mapper}, source=multilinesource1)
p.multi_line(xs='xs',ys='ys',line_color={'field' : 'sentiment', 'transform' : mapper}, source=multilinesource2)
p.text(x = 'x',y = 'y',text_color={'field':'sentiment', 'transform':mapper},text='words',text_font = 'courier', text_font_size='fontsize', source=wordsource_up, text_align='center',text_baseline='middle', name='text_up')
p.text(x = 'x',y = 'y',text_color={'field':'sentiment', 'transform':mapper},text='words',text_font = 'courier', text_font_size='fontsize', source=wordsource_down, text_align='center',text_baseline='middle', name='text_down')
p.line(x=100+spacing*breakpoint_2021 + spacing/2, y=[-300,300], line_width=5)

tdf = pd.DataFrame()
tdf['text']=['20 January 2014', '21 January 2014']
tdf['x'] = [100+spacing*breakpoint_2021 + spacing/2 - 100,100+spacing*breakpoint_2021 + spacing/2 + 100]
tdf['y'] = [-300,300]
tsource = ColumnDataSource(tdf)
p.text(x = 'x',y='y',text='text',text_font = 'courier', text_font_size='18pt', text_color="white", text_align='center',text_baseline='middle', source=tsource)
p.background_fill_color='black'
ticker = FixedTicker(ticks=[-1, 0, 1])
color_bar = ColorBar(
    title='Sentiment from negative to positive'
    ,title_text_font_style='bold'
    ,color_mapper=mapper
    ,orientation='horizontal'
    ,location=(100, 0)
    ,width=500
    ,ticker=ticker
    ,major_label_overrides = {
        1:'Positive'
        ,0:'Neutral'
        ,-1:'Negative'
    }
    ,major_label_text_font_style='normal'
    ,major_label_text_font_size='10pt'
    ,margin=100
)
p.add_layout(color_bar, 'below')
p.xgrid.grid_line_color = None
p.ygrid.grid_line_color = None

p.add_tools(HoverTool(tooltips = [
	('Full text', '@full_text')
	,('Sentiment', '@sentiment')
]
	,renderers =[p.select('circ_up')[0]]
))

p.add_tools(HoverTool(tooltips = [
	('Full text', '@full_text')
	,('Sentiment', '@sentiment')
]
	,renderers =[p.select('circ_down')[0]]
))


def update1(attr,old,new):
	entity = multi_select1.value
	print(type(entity[0]))
	newdf_up = make_df(entity_bag_2021, average_sentiment_array, entity[0], 1, spacing)
	print(newdf_up.columns)
	newsource = ColumnDataSource(newdf_up)
	wordsource_up.data=newsource.data

def update2(attr,old,new):
	entity = multi_select2.value
	newdf_down = make_df(entity_bag_2021, average_sentiment_array, entity[0], -1, spacing)
	newsource = ColumnDataSource(newdf_down)
	wordsource_down.data=newsource.data

menu = [(ent_key[key],key) for key in ent_key.keys()]
multi_select1 = MultiSelect(title="Entity type", value=["PERSON", "person"],options=menu)
multi_select1.on_change('value',update1)

multi_select2 = MultiSelect(title="Entity type", value=["ORG","Organization"],options=menu)
multi_select2.on_change('value',update2)

# ipdb.set_trace()

pre = PreText(text="""Select article by clicking on circles""",
width=500, height=100)

p.tools.append(TapTool(plot=p))
def handler(attr, old, new):
	print('attr: {} old: {} new: {}'.format(attr, old, new))
	if new['1d']['indices']:
		art_index = new['1d']['indices'][0]
		print("article index",art_index)
		file = open("MC1 Data/MC1 Data/articles/"+str(article_2021[art_index]) + ".txt","r",encoding='ISO-8859-1')
		text = file.read()
		text = text + "\n"
	else:
		text = "Select article by clicking on circles"
	pre.text=text
	

circlesource.on_change('selected', handler)

# taptool = p.select(type=TapTool)
# taptool.callback = tap_update()





spacing = 120
worddf_up = make_df(entity_bag_hist,average_sentiment_array_hist,'PERSON',1,spacing)
worddf_down = make_df(entity_bag_hist,average_sentiment_array_hist,'ORG',-1,spacing)

wordsource_up_hist = ColumnDataSource(worddf_up)
wordsource_down_hist = ColumnDataSource(worddf_down)

linesource = ColumnDataSource({
		'x' : pd.Series([(100 + spacing*i) for i in range(0,len(article_hist))])
		,'sentiment' : pd.Series(average_sentiment_array_hist)
	})

circlesource2 = ColumnDataSource({
		'x' : pd.Series([(100 + spacing*i) for i in range(0,len(article_hist))])
		,'sentiment' : pd.Series(average_sentiment_array_hist)
		,'date' : date_list
	})

xs = [[(100 + spacing*i),(100 + spacing*i)] for i in range(0,len(article_hist))]
ys = [[0,60] for i in range(0,len(article_hist))]

multilinesource1 = ColumnDataSource({
		'xs' : xs,
		'ys' : ys,
		'sentiment' : average_sentiment_array_hist
	})
multilinesource2 = ColumnDataSource({
		'xs' : xs,
		'ys' : [[0,-1*60] for i in range(0,len(article_hist))],
		'sentiment' : average_sentiment_array_hist
	})

colormap = cm.get_cmap("RdYlGn")
bokehpalette = [mpl.colors.rgb2hex(m) for m in colormap(np.arange(colormap.N))]
mapper = LinearColorMapper(palette=bokehpalette, low=-1.0, high=1.0)


tools='box_zoom, wheel_zoom, pan, reset, xwheel_pan'
p2 = figure(tools=tools, plot_width=1000, plot_height=800, title="News articles clustering based on bag of words overlap",y_range=(-400,400),x_range=(0,800))

p2.line(x='x', y=0, line_width=2, source=linesource)
p2.circle(x='x',y=0,radius=5,color={'field':'sentiment', 'transform':mapper},source=circlesource2,name='cr_hist')
p2.circle(x = 'x',y = 'y',color='black', radius='circ_rad', source=wordsource_up_hist, name='circ_up_hist')
p2.circle(x = 'x',y = 'y',color='black', radius='circ_rad', source=wordsource_down_hist, name='circ_down_hist')
p2.multi_line(xs='xs',ys='ys',line_color={'field' : 'sentiment', 'transform' : mapper}, source=multilinesource1)
p2.multi_line(xs='xs',ys='ys',line_color={'field' : 'sentiment', 'transform' : mapper}, source=multilinesource2)
p2.text(x = 'x',y = 'y',text_color={'field':'sentiment', 'transform':mapper},text='words',text_font = 'courier', text_font_size='fontsize', source=wordsource_up_hist, text_align='center',text_baseline='middle')
p2.text(x = 'x',y = 'y',text_color={'field':'sentiment', 'transform':mapper},text='words',text_font = 'courier', text_font_size='fontsize', source=wordsource_down_hist, text_align='center',text_baseline='middle')

p2.background_fill_color='black'
ticker2 = FixedTicker(ticks=[-1, 0, 1])
color_bar2 = ColorBar(
    title='Sentiment from negative to positive'
    ,title_text_font_style='bold'
    ,color_mapper=mapper
    ,orientation='horizontal'
    ,location=(100, 0)
    ,width=500
    ,ticker=ticker2
    ,major_label_overrides = {
        1:'Positive'
        ,0:'Neutral'
        ,-1:'Negative'
    }
    ,major_label_text_font_style='normal'
    ,major_label_text_font_size='10pt'
    ,margin=100
)
p2.add_layout(color_bar2, 'below')
p2.xgrid.grid_line_color = None
p2.ygrid.grid_line_color = None

p2.add_tools(HoverTool(tooltips = [
	('Full text', '@full_text')
	,('Sentiment', '@sentiment')
]
	,renderers =[p2.select('circ_up_hist')[0]]
))

p2.add_tools(HoverTool(tooltips = [
	('Full text', '@full_text')
	,('Sentiment', '@sentiment')
]
	,renderers =[p2.select('circ_down_hist')[0]]
))

p2.add_tools(HoverTool(tooltips = [
	('date', '@date')
]
	,renderers =[p2.select('cr_hist')[0]]
))

def update3(attr,old,new):
	entity = multi_select3.value
	print(type(entity[0]))
	newdf_up = make_df(entity_bag_hist, average_sentiment_array_hist, entity[0], 1, spacing)
	print(newdf_up.columns)
	newsource = ColumnDataSource(newdf_up)
	wordsource_up_hist.data=newsource.data

def update4(attr,old,new):
	entity = multi_select4.value
	newdf_down = make_df(entity_bag_hist, average_sentiment_array_hist, entity[0], -1, spacing)
	newsource = ColumnDataSource(newdf_down)
	wordsource_down_hist.data=newsource.data

menu = [(ent_key[key],key) for key in ent_key.keys()]
multi_select3 = MultiSelect(title="Entity type", value=["PERSON", "person"],options=menu)
multi_select3.on_change('value',update3)

multi_select4 = MultiSelect(title="Entity type", value=["ORG","Organization"],options=menu)
multi_select4.on_change('value',update4)

pre2 = PreText(text="""Select article by clicking on circles""",
width=500, height=100)

p2.tools.append(TapTool(plot=p2))
def handler2(attr, old, new):
	print('attr: {} old: {} new: {}'.format(attr, old, new))
	if new['1d']['indices']:
		art_index = new['1d']['indices'][0]
		print("article index",art_index)
		file = open("MC1 Data/MC1 Data/articles/"+str(article_hist[art_index]) + ".txt","r",encoding='ISO-8859-1')
		text = file.read()
		text = text + "\n"
	else:
		text="Select article by clicking on circles"
	pre2.text=text
	

circlesource2.on_change('selected', handler2)

curdoc().add_root(column(row(p,column(multi_select1,multi_select2,pre)), row(p2,column(multi_select3,multi_select4,pre2))))




