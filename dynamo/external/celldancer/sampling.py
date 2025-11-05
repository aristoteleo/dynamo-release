import pandas as pd
import numpy as np
from numpy.core.fromnumeric import size
import scipy
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt


def sampling_neighbors(gene_unsplice_splice,step=(30,30),percentile=25):

    from scipy.stats import norm
    def gaussian_kernel(X, mu = 0, sigma=1):
        return np.exp(-(X - mu)**2 / (2*sigma**2)) / np.sqrt(2*np.pi*sigma**2)
    grs = []
    for dim_i in range(gene_unsplice_splice.shape[1]):
        m, M = np.min(gene_unsplice_splice[:, dim_i]), np.max(gene_unsplice_splice[:, dim_i])
        m = m - 0.025 * np.abs(M - m)
        M = M + 0.025 * np.abs(M - m)
        gr = np.linspace(m, M, step[dim_i])
        grs.append(gr)
    meshes_tuple = np.meshgrid(*grs)
    gridpoints_coordinates = np.vstack([i.flat for i in meshes_tuple]).T
    gridpoints_coordinates = gridpoints_coordinates + norm.rvs(loc=0, scale=0.15, size=gridpoints_coordinates.shape)
    
    np.random.seed(10) # set random seed
    
    nn = NearestNeighbors()

    neighbors_1 = min((gene_unsplice_splice[:,0:2].shape[0]-1), 20)
    nn.fit(gene_unsplice_splice[:,0:2])
    dist, ixs = nn.kneighbors(gridpoints_coordinates, neighbors_1)

    ix_choice = ixs[:,0].flat[:]
    ix_choice = np.unique(ix_choice)

    nn = NearestNeighbors()

    neighbors_2 = min((gene_unsplice_splice[:,0:2].shape[0]-1), 20)
    nn.fit(gene_unsplice_splice[:,0:2])
    dist, ixs = nn.kneighbors(gene_unsplice_splice[ix_choice, 0:2], neighbors_2)
    
    density_extimate = gaussian_kernel(dist, mu=0, sigma=0.5).sum(1)
    bool_density = density_extimate > np.percentile(density_extimate, percentile)
    ix_choice = ix_choice[bool_density]
    return(ix_choice)

def sampling_inverse(gene_unsplice_splice,target_amount=500):
    unsplice = gene_unsplice_splice[:,0]
    splice = gene_unsplice_splice[:,1]
    values = np.vstack([unsplice,splice])
    kernel = scipy.stats.gaussian_kde(values)
    p = kernel(values)
    # p2 = (1/p)/sum(1/p)
    p2 = (1/p)/sum(1/p)
    idx = np.arange(values.shape[1])
    r = scipy.stats.rv_discrete(values=(idx, p2))
    idx_choice = r.rvs(size=target_amount)
    return(idx_choice)

def sampling_circle(gene_unsplice_splice,target_amount=500):
    unsplice = gene_unsplice_splice[:,0]
    splice = gene_unsplice_splice[:,1]
    values = np.vstack([unsplice,splice])
    kernel = scipy.stats.gaussian_kde(values)
    p = kernel(values)
    idx = np.arange(values.shape[1])
    tmp_p = np.square((1-(p/(max(p)))**2))+0.0001
    p2 = tmp_p/sum(tmp_p)
    r = scipy.stats.rv_discrete(values=(idx, p2))
    idx_choice = r.rvs(size=target_amount)
    return(idx_choice)

def sampling_random(gene_unsplice_splice, target_amount=500):
    idx = np.random.choice(gene_unsplice_splice.shape[0], size = target_amount, replace=False)
    return(idx)
    
def sampling_adata(detail, 
                    para,
                    target_amount=500,
                    step=(30,30)):
    if para == 'neighbors':
        data_U_S= np.array(detail[["unsplice","splice"]])
        idx = sampling_neighbors(data_U_S,step)
    elif para == 'inverse':
        data_U_S= np.array(detail[["unsplice","splice"]])
        idx = sampling_inverse(data_U_S,target_amount)
    elif para == 'circle':
        data_U_S= np.array(detail[["unsplice","splice"]])
        idx = sampling_circle(data_U_S,target_amount)
    elif para == 'random':
        data_U_S= np.array(detail[["unsplice","splice"]])
        idx = sampling_random(data_U_S,target_amount)
    else:
        print('para is neighbors or inverse or circle')
    return(idx)

def sampling_embedding(detail, 
                    para,
                    target_amount=500,
                    step=(30,30)):

    '''
    Guangyu
    '''
    if para == 'neighbors':
        data_U_S= np.array(detail[["embedding1","embedding2"]])
        idx = sampling_neighbors(data_U_S,step)
    elif para == 'inverse':
        print('inverse')
        data_U_S= np.array(detail[["embedding1","embedding2"]])
        idx = sampling_inverse(data_U_S,target_amount)
    elif para == 'circle':
        data_U_S= np.array(detail[["embedding1","embedding2"]])
        idx = sampling_circle(data_U_S,target_amount)
    elif para == 'random':
        # print('random')
        data_U_S= np.array(detail[["embedding1","embedding2"]])
        idx = sampling_random(data_U_S,target_amount)
    else:
        print('para is neighbors or inverse or circle')
    return(idx)

def adata_to_detail(data, para, gene):
    '''
    convert adata to detail format
    data: an anndata
    para: the varable name of unsplice, splice, and gene name
    para = ['Mu', 'Ms']
    '''
    data2 = data[:, data.var.index.isin([gene])].copy()
    unsplice = data2.layers[para[0]][:,0].copy().astype(np.float32)
    splice = data2.layers[para[1]][:,0].copy().astype(np.float32)
    detail = pd.DataFrame({'gene_name':gene, 'unsplice':unsplice, 'splice':splice})
    return(detail)

def downsampling_embedding(data_df,para,target_amount, step, n_neighbors,expression_scale=None,projection_neighbor_choice='embedding',pca_n_components=None,umap_n=None,umap_n_components=None):
    '''
    Guangyu
    sampling cells by embedding
    data—df: from load_cellDancer
    para:
    
    return: sampled embedding, the indexs of sampled cells, and the neighbors of sampled cells
    '''

    gene = data_df['gene_name'].drop_duplicates().iloc[0]
    embedding = data_df.loc[data_df['gene_name']==gene][['embedding1','embedding2']]

    if step is not None:
        idx_downSampling_embedding = sampling_embedding(embedding,
                    para=para,
                    target_amount=target_amount,
                    step=step)
    else:
        idx_downSampling_embedding=range(0,embedding.shape[0]) # all cells
        
    def transfer(data_df,expression_scale):
        if expression_scale=='log':
            data_df.splice=np.log(data_df.splice+0.000001)
            data_df.unsplice=np.log(data_df.unsplice+0.000001)
        elif expression_scale=='2power':
            data_df.splice=2**(data_df.splice)
            data_df.unsplice=2**(data_df.unsplice)
        elif expression_scale=='power10':
            data_df.splice=(data_df.splice)**10
            data_df.unsplice=(data_df.unsplice)**10
        elif expression_scale=='2power_norm_multi10':
            gene_order=data_df.gene_name.drop_duplicates()
            onegene=data_df[data_df.gene_name==data_df.gene_name[0]]
            cellAmt=len(onegene)
            data_df_max=data_df.groupby('gene_name')[['splice','unsplice']].max().rename(columns={'splice': 'splice_max','unsplice': 'unsplice_max'})
            data_df_min=data_df.groupby('gene_name')[['splice','unsplice']].min().rename(columns={'splice': 'splice_min','unsplice': 'unsplice_min'})
            data_df_fin=pd.concat([data_df_max,data_df_min],axis=1).reindex(gene_order)
            data_df_fin=data_df_fin.loc[data_df_fin.index.repeat(cellAmt)]
            data_df_combined=pd.concat([data_df.reset_index(drop=True) ,data_df_fin[['splice_max','unsplice_max','splice_min','unsplice_min']].reset_index(drop=True)],axis=1)
            data_df_combined['unsplice_norm']=''
            data_df_combined['splice_norm']=''
            data_df_combined.unsplice_norm=(data_df_combined.unsplice-data_df_combined.unsplice_min)/(data_df_combined.unsplice_max-data_df_combined.unsplice_min)
            data_df_combined.splice_norm=(data_df_combined.splice-data_df_combined.splice_min)/(data_df_combined.splice_max-data_df_combined.splice_min)
            data_df_combined.unsplice=2**(data_df_combined.unsplice_norm*10)
            data_df_combined.splice=2**(data_df_combined.splice_norm*10)
            data_df=data_df_combined

        return (data_df)

    data_df=transfer(data_df,expression_scale)
    

    if projection_neighbor_choice=='gene':
        #print('using gene projection_neighbor_choice')
        cellID = data_df.loc[data_df['gene_name']==gene]['cellID']
        data_df_pivot=data_df.pivot(index='cellID', columns='gene_name', values='splice').reindex(cellID)
        embedding_downsampling = data_df_pivot.iloc[idx_downSampling_embedding]
    elif projection_neighbor_choice=='pca': # not use
        from sklearn.decomposition import PCA
        #print('using pca projection_neighbor_choice')
        cellID = data_df.loc[data_df['gene_name']==gene]['cellID']
        data_df_pivot=data_df.pivot(index='cellID', columns='gene_name', values='splice').reindex(cellID)
        embedding_downsampling_0 = data_df_pivot.iloc[idx_downSampling_embedding]
        pca=PCA(n_components=pca_n_components)
        pca.fit(embedding_downsampling_0)
        embedding_downsampling = pca.transform(embedding_downsampling_0)[:,range(pca_n_components)]
    elif projection_neighbor_choice=='pca_norm':
        from sklearn.decomposition import PCA
        #print('pca_norm')
        cellID = data_df.loc[data_df['gene_name']==gene]['cellID']
        data_df_pivot=data_df.pivot(index='cellID', columns='gene_name', values='splice').reindex(cellID)
        embedding_downsampling_0 = data_df_pivot.iloc[idx_downSampling_embedding]
        pca=PCA(n_components=pca_n_components)
        pca.fit(embedding_downsampling_0)
        embedding_downsampling_trans = pca.transform(embedding_downsampling_0)[:,range(pca_n_components)]
        embedding_downsampling_trans_norm=(embedding_downsampling_trans - embedding_downsampling_trans.min(0)) / embedding_downsampling_trans.ptp(0)#normalize
        embedding_downsampling_trans_norm_mult10=embedding_downsampling_trans_norm*10 #optional
        embedding_downsampling=embedding_downsampling_trans_norm_mult10**5 # optional
    elif projection_neighbor_choice=='embedding':
        embedding_downsampling = embedding.iloc[idx_downSampling_embedding][['embedding1','embedding2']]

    elif projection_neighbor_choice =='umap':
        import umap
        #print('using umap projection_neighbor_choice')
        cellID = data_df.loc[data_df['gene_name']==gene]['cellID']
        data_df_pivot=data_df.pivot(index='cellID', columns='gene_name', values='splice').reindex(cellID)
        embedding_downsampling_0 = data_df_pivot.iloc[idx_downSampling_embedding]
        
        def get_umap(df,n_neighbors=umap_n, min_dist=0.1, n_components=umap_n_components, metric='euclidean'): 
            fit = umap.UMAP(
                n_neighbors=n_neighbors,
                min_dist=min_dist,
                n_components=n_components,
                metric=metric
            )
            embed = fit.fit_transform(df);
            return(embed)
        embedding_downsampling=get_umap(embedding_downsampling_0)

    n_neighbors = min(int((embedding_downsampling.shape[0])/4), n_neighbors)
    if n_neighbors==0:
        n_neighbors=1
    nn = NearestNeighbors(n_neighbors=n_neighbors)
    nn.fit(embedding_downsampling) 
    embedding_knn = nn.kneighbors_graph(mode="connectivity")
    return(embedding_downsampling, idx_downSampling_embedding, embedding_knn)

def downsampling(data_df, gene_list, downsampling_ixs):
    '''
    Guangyu
    '''
    data_df_downsampled=pd.DataFrame()
    for gene in gene_list:
        data_df_one_gene=data_df[data_df['gene_name']==gene]
        data_df_one_gene_downsampled = data_df_one_gene.iloc[downsampling_ixs]
        data_df_downsampled=data_df_downsampled.append(data_df_one_gene_downsampled)
    return(data_df_downsampled)
