import qaoa_methods
import operator_pools
import networkx as nx
import numpy as np

n = 6
p = 10
g = nx.Graph()
g.add_nodes_from(np.arange(0, n, 1))
elist = [(0, 1, 0.304082), (0, 5, 0.480569), (0, 3, 0.000471), (1, 2, 0.282721), (1, 4, 0.883705), (2, 3, 0.749396), (2, 5, 0.89423), (3, 4, 0.665313), (4, 5, 0.0211785)]
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