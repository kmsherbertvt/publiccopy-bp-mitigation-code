import qaoa_methods
import operator_pools
import networkx as nx
import numpy as np

n = 6
p = 10
g = nx.Graph()
g.add_nodes_from(np.arange(0, n, 1))
elist = [(0, 1, 0.374015), (0, 5, 0.956849), (0, 3, 0.141309), (1, 2, 0.013096), (1, 4, 0.435166), (2, 3, 0.931134), (2, 5, 0.762881), (3, 4, 0.324137), (4, 5, 0.904244)]
g.add_weighted_edges_from(elist)

filename = 'error' + '.txt'
filename_1 = 'mix_operator_' + '.txt'
f = open(filename, "a")
f_mix = open(filename_1, "a")
qaoa_methods.run(n, 
	             g, 
	             f,
	             f_mix,
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