colors_alpha_beta_gamma = ["#007EB7","#3B9AB2", "#78B7C5", "#EBCC2A", "#E1AF00", "#F21A00"] 
colors_splice_unsplice = ["#2488F0","#7F3F98","#E22929","#FCB31A"]

colormap_erythroid={
'Haematoendothelial progenitors':'#3361A5',
'Blood progenitors 1':'#248AF3',
'Blood progenitors 2':'#14B3FF',
'Erythroid1':'#88CEEF',
'Erythroid2':'#FDB31A',
'Erythroid3':'#E42A2A'
}

colormap_neuro = {
'CA': "#ed0345",
'CA1-Sub': "#710162",
'CA2-3-4': "#a12a5e",
'Granule':"#ef6a32",
'ImmGranule1': "#ef6a32",
'ImmGranule2': "#ef6a32",
'Nbl1': "#fbbf45",
'Nbl2': "#fbbf45",
'nIPC': "#aad962",
'RadialGlia': "#03c383",
'RadialGlia2': "#03c383",
'GlialProg': '#56A65A',
'OPC': "#017351",
'ImmAstro': "#08A8CE"
}


colormap_pancreas={
'Ductal':'#3361A5',
'Ngn3 low EP':'#248AF3',
'Ngn3 high EP':'#14B3FF',
'Pre-endocrine':'#88CEEF',
'Alpha':'#ff4800',
'Beta':"#B81136",
'Delta':'green',
'Epsilon':'#03B3B0'
}

colormap_hgForebrainGlut={
0:'#9408F7',
1:'#C729D6',
2:'#FA4AB5',
3:'#FF6A95',
4:'#FF8B74',
5:'#FFAC53',
6:'#FFCD32'
}

colormap_hgforebrainglut={
'Radial Glia':'#9408F7',
'Radial Glia':'#C729D6',
'Neuroblast':'#FA4AB5',
'Neuroblast':'#FF6A95',
'Immature Neuron':'#FF8B74',
'Immature Neuron':'#FFAC53',
'Neuron':'#FFCD32'
}

color_template = ["#08A8CE","#017351",'#56A65A',"#03c383","#aad962","#fbbf45","#ef6a32","#ed0345","#a12a5e","#710162","#3B9AB2"]

def build_colormap(cluster_list):
    from itertools import cycle
    color_list=color_template
    colors = dict(zip(cluster_list, cycle(color_list)) if len(cluster_list) > len(color_list) else zip(cycle(cluster_list), color_list))
    return colors

