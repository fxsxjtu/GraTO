#encoding:utf-8
from collections import namedtuple
#Genotype = namedtuple('Genotype', 'normal blockwise')
Genotype = namedtuple('Genotype', 'normal normal_concat')
PRIMITIVES = [
    #'none',
    'GCN_norm',
    'GAT_norm',
    #'skip_connect',
    'GraphSAGE_norm',
    'AGNN_norm',
    'SGC_norm',
    'Drop_attr3',
    'Drop_attr4',
    'Drop_attr5',
    'Drop_Edge',
]
DARTS_pubmed=Genotype(normal=[('GCN_norm', 1),('GCN_norm', 0),('AGNN_norm', 1),('Drop_attr5', 2),('GraphSAGE_norm', 1),('Drop_attr3', 0),('Drop_attr4', 2),('GCN_norm', 3)],normal_concat=range(2, 6))# if you find a new structure, just follow this form and add your structure.
