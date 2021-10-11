import pickle

network_inference_genes = [
    "GATA2",
    "GFI1",
    "GFI1B",
    "NFE2",
    "TAL1",
    "GATA1",
    "LYL1",
    "ERG",
    "FLI1",
    "MEIS1",
    "SPI1",
    "MITF",
    "LMO2",
    "LDB1",
    "RUNX1",
    "ETV6",
    "HHEX",
    "CBFA2T3",
]
Krumsiek_11 = ["GATA2", "GATA1", "ZFPM1", "SPI1", "FLI1", "KLF1", "TAL1", "CEBPA", "GFI1", "JUN", "NAB2"]

tmp2 = pickle.load(open("/Users/xqiu/Downloads/gene2celltype.p", "rb"))
len(tmp2.keys()) + len(network_inference_genes + Krumsiek_11)


print(ranking.query("TF == True").shape)
