import qaoa_methods
import operator_pools
import networkx as nx
import numpy as np

n = 6
p = 8
g = nx.Graph()
g.add_nodes_from(np.arange(0, n, 1))
elist = [(0, 1, 0.516887), (0, 5, 0.553098), (0, 3, 0.289658), (1, 2, 0.485823), (1, 4, 0.493809), (2, 3, 0.141917), (2, 5, 0.842542), (3, 4, 0.844711), (4, 5, 0.529479)]
g.add_weighted_edges_from(elist)

filename = 'error' + '.txt'
filename_1 = 'mix_operator' + '.txt'
filename_2 = 'parameter' + '.txt'
f = open(filename, "a")
f_mix = open(filename_1, "a")
f_para = open(filename_2, "a")
qaoa_methods.run(n, 
	             g, 
	             f,
	             f_mix,
	             f_para,
	             adapt_thresh=1e-10, 
	             theta_thresh=1e-7,
	             layer=p, 
	             pool=operator_pools.qaoa(), 
	             init_para=0.01, 
	             structure = 'qaoa', 
	             selection = 'grad', 
	             rand_ham = 'False', 
	             opt_method = 'NM',
	             landscape = False,
	             landscape_after = False,
	             resolution = 100)
f.close()
f_mix.close()
f_para.close()