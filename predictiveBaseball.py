import numpy as np
import pandas as pd
from pybaseball import batting_stats_bref, bwar_bat
import csv
import math
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE


def get_features(year, advanced=False):
	'''

	This function gathers the features either using advanced stats or traditional stats
	based on the input given by the user as well as based on the year given by the user.
	
	'''
	#----------------------------------------------------------------------

	'''

	The next block of code contains helper functions that will either get both the advanced stats
	and the traditional stats that will be used as features. There is also a helper function to
	scale the features.

	'''

	#----------------------------------------------------------------------

	def scale(features):
		scaler=StandardScaler()
		feature_scaler=scaler.fit(features)
		return feature_scaler.fit_transform(features)

	def get_traditional_stats(year):
		stats=batting_stats_bref(year)
		xbh=(stats['2B']+stats['3B']+stats['HR'])/stats['G']
		stealAttempts=(stats['SB']+stats['CS'])/stats['G']

		features=pd.concat([xbh, stealAttempts], axis=1)

		feature_list=['R', 'H', 'RBI', 'BB', 'IBB', 'SH', 'SF', 'SO']

		for feature in feature_list:
			features=pd.concat([features, stats[feature]], axis=1)

		return scale(features)

	def get_advanced_stats(year):
		stats=bwar_bat(True)
		year=stats['year_ID'] == year
		batter=stats['pitcher']=='N'
		stint=stats['stint_ID']==1

		stats=stats.loc[year & batter & stint]

		features=stats['age']

		feature_list=['runs_bat', 'runs_br', 'runs_dp', 'runs_field', 'runs_infield', 'runs_outfield', 'runs_position', 'runs_defense', 'runs_above_avg_off', 'runs_above_avg_def', 'OPS_plus', 'runs_good_plays']

		for feature in feature_list:
			features=pd.concat([features, stats[feature]], axis=1)

		return scale(features.fillna(value=0))
	#--------------------------------------------------------------------------

	'''

	The next block either returns the advanced features or the traditional features.

	'''

	#--------------------------------------------------------------------------

	if advanced:
		return get_advanced_stats(year)
	else:
		return get_traditional_stats(year)

def train(year, advanced=False, visualize=True, epsilon=None):
	#--------------------------------------------------------

	'''
	This function trains the feature set using kmeans clustering or visualizes the 
	n dimensional data in two dimensions. The default is to use non-advanced statistics
	and to simply visualize the dataset, but the user can view the amount of clusters if
	they wish. Only requires a year inputted to run.
	
	'''

	#--------------------------------------------------------
	print('Creating Feature Set...')
	features=get_features(year, advanced)
	print('Feature Set Created!')

	print('Creating Training Set...')
	train, _ = train_test_split(features, test_size=0.2)
	print('Training Set created!')

	if visualize:
		#-----------------------------------------------------

		'''

		If the user wants to visualize the data in a 2-Dimensional
		environment, this will reduce the dimensionality and allow
		for viewing

		'''

		#----------------------------------------------------
		print('Visualizing Data...')
		train_embedded=TSNE(n_components=2).fit_transform(train)
		plt.scatter(train_embedded[:, 0],train[:, 1])
		plt.show()
	else:
		#-----------------------------------------------------

		'''
	
		If the user wants to see the locations of the clusters in n-dimensional 
		space and which cluster has the lowest sum of squared distances of points
		to their closest cluster center (the best amount of clusters)


		'''

		#-----------------------------------------------------
		inertia=[]
		clusters=[]
		addCluster=True
		count=0
		for n in range(2,20):
			kmeans = KMeans(n_clusters=n, random_state=0).fit(train)
			for cluster1 in kmeans.cluster_centers_:
				for cluster2 in kmeans.cluster_centers_:
					if set(cluster1)!=set(cluster2):
						if sum(map(lambda x:x*x,cluster1-cluster2))<epsilon:
							addCluster=False
			if addCluster:
				print('KMeans Clustering for '+str(n)+' clusters')
				clusters.append(kmeans.cluster_centers_)
				print(clusters[count])
				print(kmeans.inertia_)
				inertia.append(kmeans.inertia_)
				count+=1
		print('The best number of clusters is '+ str(inertia.index(min(inertia))+2)+ ' clusters. The samples are '+ str(min(inertia))+ ' apart from their cluster center in terms of squared distance.')

train(year=2017)




