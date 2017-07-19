import pandas as pd 
from bokeh.plotting import figure
from bokeh.io import output_file,show,curdoc
from bokeh.models import CategoricalColorMapper,ColumnDataSource,HoverTool
from bokeh.palettes import Spectral6
from bokeh.layouts import row,widgetbox,column
from bokeh.models.widgets import Slider,Select

import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


data=pd.read_csv('winequality-red.csv',sep=';')
data=data.set_index('quality')
alcohol_range = data.alcohol.unique().tolist()
source=ColumnDataSource(data={'x':data['volatile acidity'].loc[3],
							  'y':data['residual sugar'].loc[3],
							  'alcohol':data['alcohol'].loc[3]})
p=figure(plot_width=400,plot_height=400,x_axis_label='volatile acidity',y_axis_label='residual sugar')
mapper=CategoricalColorMapper(factors=alcohol_range,palette=Spectral6)
p.asterisk(x='x',y='y',size=10,source=source,color={'field':'alcohol','transform':mapper},legend='alcohol')
p.legend.location='top_right'
p.title.text='Wine Data'


def update_plot(attr,old,new):
	qua=slider.value
	x=x_select.value
	y=y_select.value
	p.xaxis.axis_label=x
	p.yaxis.axis_label=y
	new_data={'x':data[x].loc[qua],'y':data[y].loc[qua],'alcohol':data['alcohol'].loc[qua]}
	source.data=new_data

slider=Slider(start=3,end=8,value=3,step=1,title='Wine Quality')
slider.on_change('value',update_plot)

x_select=Select(options=['volatile acidity','fixed acidity','citric acid'],value='volatile acidity',title='x-axis data')
x_select.on_change('value',update_plot)
y_select=Select(options=['residual sugar','density','pH'],value='residual sugar',title='y-axis data')
y_select.on_change('value',update_plot)


#classify

df=pd.read_csv('winequality-red.csv',sep=';')

quality=df['quality']
wine=df.drop('quality',axis=1)
wine_train,wine_test,quality_train,quality_test=train_test_split(wine,quality,test_size=0.3,random_state=42,stratify=quality)

knn = KNeighborsClassifier(n_neighbors=1 )
knn.fit(wine_train,quality_train)

svc=SVC()
svc.fit(wine_train,quality_train)

dt=DecisionTreeClassifier()
dt.fit(wine_train,quality_train)
quality_pred=knn.predict(wine_test)

hover=HoverTool(tooltips=([('actual','@quality_test'),('predicted','@quality_pred')]))
c_source=ColumnDataSource(data={'x':quality_test,'y':quality_pred})
c=figure(plot_width=400,plot_height=400,x_axis_label='Actual Wine Quality',y_axis_label='Predicted Wine Quality')
c.circle(x='x',y='y',source=c_source)
c.add_tools(hover)
def callback(attr,old,new):
	if select.value=='svm':
		quality_pred=svc.predict(wine_test)
		c_source.data={'x':quality_test,'y':quality_pred}
	elif select.value=='decision tree':
		quality_pred=dt.predict(wine_test)
		c_source.data={'x':quality_test,'y':quality_pred}
	else:
		quality_pred=knn.predict(wine_test)
		c_source.data={'x':quality_test,'y':quality_pred}
select=Select(options=['knn','svm','decision tree'],value='knn',title='Classifier')
select.on_change('value',callback)

layout=row(row(widgetbox(slider,x_select,y_select),p),widgetbox(select),c)
curdoc().add_root(layout)	