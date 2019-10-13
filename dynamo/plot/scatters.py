import numpy as np
import pandas as pd 
from sklearn.neighbors import NearestNeighbors
from scipy.stats import norm


def scatter(adata, basis, color, size): # we can also use the color to visualize the velocity 
   from plotnine import ggplot, geom_scatter
   X=adata.obsm['X_' + basis]
   df=pd.DataFrame(X)
   df['color_group'] = adata.obs[:, color]
   ggplot(X[0], X[1], color=df['color_group']) + geom_scatter()


def mean_cv(adata, group): 
	from plotnine import ggplot, geom_scatter
	mn = adata.obs['mean']
	cv = adata.obs['cv']
	df = pd.DataFrame({'mean': mn, 'cv': CV})

	ggplot(mean, cv, data=df) + geom_scatter()


def variance_explained(adata):
	from plotnine import ggplot, geom_scatter
	var_info = adata.uns['pca']

	ggplot(variance, compoent, data = var_info) + geom_scatter()


#def velocity(adata, type) # type can be either one of the three, cellwise, velocity on grid, streamline plot. 
#	"""
#
#	"""
# 
