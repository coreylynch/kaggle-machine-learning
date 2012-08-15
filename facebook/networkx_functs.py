import networkx as nx    # using 1.6, from http://networkx.lanl.gov/ 
from networkx.readwrite import json_graph
from operator import itemgetter
import json 
import sys
import numpy as np
from pandas import DataFrame
from time import time
from collections import deque
import networkx as nx

'''
M = build_sparse_matrix('train.csv')
G = nx.from_scipy_sparse_matrix(M,create_using=nx.DiGraph())
del M
'''
##########################################################################################

#Utilities 

def read_in_edges(filename, info=True):
	''' 
	Read a directed graph by default.  Change from DiGraph to Graph below if you want undirected. 
	Edgelist looks like this:
	node1 node2
	node3 node1
	node1 node3
	 ...
	 '''
	g_orig = nx.read_edgelist(filename, create_using=nx.DiGraph())
	if info:
		print "Read in edgelist file ", filename
		print nx.info(g_orig)
	return g_orig
	
def save_to_jsonfile(filename, graph):
	'''
	Save graph object to filename 
	'''
	g = graph
	g_json = json_graph.node_link_data(g) # node-link format to serialize
	json.dump(g_json, open(filename,'w'))

def read_json_file(filename, info=True):
	'''
	Use if you already have a json rep of a graph and want to update/modify it
	'''
	graph = json_graph.load(open(filename))
	if info:
		print "Read in file ", filename
		print nx.info(graph)
	return graph
	
def report_node_data(graph, node=""):
	'''
	Will tell you what attributes exist on nodes in the graph.
	Useful for checking your updates.
	'''
	
	g = graph
	if len(node) == 0:
		print "Found these sample attributes on the nodes:"
		print g.nodes(data=True)[0]
	else:
		print "Values for node " + node
		print [d for n,d in g.nodes_iter(data=True) if n==node]

# 3 different function wrappers 
# source: http://www.islab.ntua.gr/attachments/article/63/06033365.pdf
def score1(func,graph,a,b):
	return(func(graph,a,b))

def score2(func,graph,a,b):
	a_outbound = graph[a]
	scores = np.array([func(graph,b,i) for i in a_outbound])
	try:
		return (1 / float(len(a_outbound)) * np.sum(scores))
	except ZeroDivisionError:
		return 0

def score3(func,graph,graph_inverse,a,b):
	b_inbound = graph_inverse[b]
	scores = np.array([func(graph,a,i) for i in b_inbound])
	try:
		return (1 / float(len(b_inbound)) * np.sum(scores))
	except ZeroDivisionError:
		return 0


def common_friends_out(graph,a,b):
    outs_a = graph[a]
    outs_b = graph[b]
    return len(set(outs_a).intersection(outs_b))

def common_friends_in(graph_inverse,a,b):
    ins_a = graph_inverse[a]
    ins_b = graph_inverse[b]
    return len(set(ins_a).intersection(ins_b))

def common_friends_bi(graph,graph_inverse,a,b):
    outs_a = graph[a]
    ins_a = graph_inverse[a]
    outs_b = graph[b]
    ins_b = graph_inverse[b]
    return len(set(outs_a+ins_a).intersection(outs_b+ins_b))

##########################################################################################
# Edge Features

def pagerank_product(graph,a,b):
	n1 = neighborhood(graph,a)
	n2 = neighborhood(graph,b)
	union = n1.union(n2)
	union.add(a)
	union.add(b)
	union_graph = nx.subgraph(graph,union)
	try:
		pr = nx.pagerank(union_graph)
		return pr[a]*pr[b]
	except:
		return 0

def pr(graph,a,b):
	n1 = neighborhood(graph,a)
	n2 = neighborhood(graph,b)
	union = n1.union(n2)
	union.add(a)
	union.add(b)
	union_graph = nx.subgraph(graph,union)
	pagerank = nx.pagerank(union_graph)
	return (pagerank[a], pagerank[b])

def page_ranks(graph,nodes_to_recommend):
	total_neighborhood = set()
	for i in nodes_to_recommend:
		total_neighborhood.add(i)
		if len(neighborhood(graph,i)) <= 25:
			for j in neighborhood(graph,i):
				total_neighborhood.add(j)
	subgraph = nx.subgraph(graph,list(total_neighborhood))
	try:
		pagerank = nx.pagerank(subgraph)
	except:
		pagerank = {}
	return pagerank
	

def rank_commutative_by_pagerank(graph,nodes_to_recommend):
	pagerank = page_ranks(graph,nodes_to_recommend)
	scores = sorted(zip([pagerank[i] for i in nodes_to_recommend],nodes_to_recommend),reverse=True)
	return [i[1] for i in scores]

def full_rank_by_pagerank(graph,nodes_to_recommend):
	pagerank = page_ranks(graph,nodes_to_recommend)
	scores = sorted(zip(pagerank.values(),pagerank.keys()),reverse=True)
	return [i[1] for i in scores]

def common_friends_in(graph,a,b):
	a_inbound = set(graph.predecessors(a))
	b_inbound = set(graph.predecessors(b))
	return len(a_inbound.intersection(b_inbound))

def common_friends_out(graph,a,b):
	a_outbound = set(graph.successors(a))
	b_outbound = set(graph.successors(b))
	return len(a_outbound.intersection(b_outbound))

def common_friends_bi(graph,a,b):
	a_neighbors = neighborhood(graph,a)
	b_neighbors = neighborhood(graph,b)
	return len(a_neighbors.intersection(b_neighbors))

def total_friends(graph,a,b):
	a_neighbors = neighborhood(graph,a)
	b_neighbors = neighborhood(graph,b)
	return len(a_neighbors.union(b_neighbors))

def jaccard(graph,a,b):
	return float(common_friends_bi(graph,a,b)) / total_friends(graph,a,b)

def is_reciprocal(graph,a,b):
	if graph.has_edge(b,a):
		return 1
	else:
		return 0

def transitive_friends(graph,a,b):
	a_outbound = set(graph.successors(a))
	b_inbound = set(graph.predecessors(b))
	return len(a_outbound.intersection(b_inbound))

def preferential_attachment(graph,a,b):
	a_neighbors = neighborhood(graph,a)
	b_neighbors = neighborhood(graph,b)
	return len(a_neighbors) * len(b_neighbors)

def log_pa(graph,a,b):
	return np.log(preferential_attachment(graph,a,b))

def friends_measure(graph,a,b,edges=None):
	def sigma(x,y):
		if x==y or (x,y) in edges or (y,x) in edges:
			return 1
		else:
			return 0
	if edges is not None:
		edges = edges
	else:
		edges = graph.edges()
	friends = [[sigma(i,j) for i in neighborhood(graph,a)]for j in neighborhood(graph,b)]
	return np.sum(friends)

def friends_measure(graph,graph_inverse,a,b,edges):
	def sigma(x,y):
		if x==y or (x,y) in edges or (y,x) in edges:
			return 1
		else:
			return 0
	friends = [[sigma(i,j) for i in neighborhood(graph,graph_inverse,a)]for j in neighborhood(graph,graph_inverse,b)]
	return np.sum(friends)

##########################################################################################
# Node Features

def in_degree(G,a):
	return G.in_degree(a)

def out_degree(G,a):
	return G.out_degree(a)

def shortest_path_length(graph,a,b):
	try:
		sp = nx.shortest_path_length(graph,a,b)
	except:
		sp = np.inf
	return sp

def calculate_degree(graph):
	'''
	Calculate the degree of a node and save the value as an attribute on the node. Returns the graph and the dict of degrees.
	'''
	g = graph
	deg = nx.degree(g)
	nx.set_node_attributes(g,'degree',deg)
	return g, deg

def calculate_indegree(graph):
	'''Will only work on DiGraph (directed graph)
	Saves the indegree as attribute on the node, and returns graph, dict of indegree
	'''
	g = graph
	indeg = g.in_degree()
	nx.set_node_attributes(g, 'indegree', indeg)
	return g, indeg
	
def calculate_outdegree(graph):
	'''Will only work on DiGraph (directed graph)
	Saves the outdegree as attribute on the node, and returns graph, dict of outdegree
	'''
	g = graph
	outdeg = g.out_degree()
	nx.set_node_attributes(g, 'outdegree', outdeg)
	return g, outdeg

def calculate_katz(graph):
	import scipy.linalg as ln
	g=graph
	A = nx.adj_matrix(g)
	I = A.I
	B_katz = 0.05
	# http://www.cs.utexas.edu/~yzhang/papers/osn-imc09.pdf use B_katz of 0.05 and 0.005
	Katz = ln.pinv(I-B_katz*A) - I
	with open('pickles/katz_matrix.pkl','wb') as f:
		pickle.dump(Katz,f)
	return Katz

def neighborhood(graph,node):
	preds = graph.predecessors(node)
	succs = graph.successors(node)
	return set(preds + succs)

def calculate_neighborhood(graph):
	g=graph
	neighborhood = dict()

	for node in g:
		preds = g.predecessors(node)
		succs = g.successors(node)
		total = len(set(preds + succs))
		neighborhood[node] = total
	nx.set_node_attributes(g,'neighborhood',neighborhood)
	return g, neighborhood

def calculate_indegree_density(graph,indegree=None,neighborhood=None):
	
	g = graph
	
	if indegree is None:
		_, indegree = calculate_indegree(g)
	else:
		indegree = indegree

	if neighborhood is None:
		_, neighborhood = calculate_neighborhood(g)
	else:
		neighborhood = neighborhood

	indegree_density = np.array(indegree.values()) / np.array(neighborhood.values(),dtype=np.float64)

	in_degree_density_dict = dict(zip(indegree.keys(),indegree_density))
	nx.set_node_attributes(g,'indegree_density',in_degree_density_dict)
	return g,in_degree_density_dict


def calculate_outdegree_density(graph,outdegree=None,neighborhood=None):
	
	g = graph
	
	if outdegree is None:
		_, outdegree = calculate_outdegree(g)
	else:
		outdegree = outdegree

	if neighborhood is None:
	else:
		neighborhood = neighborhood

	outdegree_density = np.array(outdegree.values()) / np.array(neighborhood.values(),dtype=np.float64)

	out_degree_density_dict = dict(zip(outdegree.keys(),outdegree_density))
	nx.set_node_attributes(g,'outdegree_density',out_degree_density_dict)
	return g,out_degree_density_dict

def calculate_bidegree_density(graph,indegree=None,outdegree=None,neighborhood=None):

	g = graph
	if indegree is None:
		_, indegree = calculate_indegree(g)
	else:
		indegree = indegree
	if outdegree is None:
		_, outdegree = calculate_outdegree(g)
	else:
		outdegree = outdegree
	if neighborhood is None:
		_, neighborhood = calculate_neighborhood(g)
	else:
		neighborhood = neighborhood

	# have to define function to be mapped BEFORE pool of workers
	def calculate_bidegree_per_node(node):
		s1 = set(g.predecessors(node))
		s2 = set(g.successors(node))
		inter = s1.intersection(s2)
		bidegree = len(inter)
		return bidegree

		#num_neighbors = len(neighborhood[node])
		#bidegree_density[node] = np.float64(bidegree)/num_neighbors

	bidegree = dict()
	for node in g:
		bidegree[node]=calculate_bidegree_per_node(node)

	bidegree_density = np.array(bidegree.values()) / np.array(neighborhood.values(),dtype=np.float64)

	bidegree_density_dict = dict(zip(bidegree.keys(),bidegree_density))
	nx.set_node_attributes(g,'bidegree_density',bidegree_density_dict)
	return g, bidegree_density_dict


	pool = Pool(processes=8)

	pool.map(calculate_bidegree_per_node,g.nodes_iter())
	return g,bidegree_density

def calculate_betweenness(graph):
	''' Calculate betweenness centrality of a node, sets value on node as attribute; returns graph, and dict of the betweenness centrality values
	'''
	g = graph
	bc=nx.betweenness_centrality(g)
	nx.set_node_attributes(g,'betweenness',bc)
	return g, bc
	
def calculate_eigenvector_centrality(graph):  
	''' Calculate eigenvector centrality of a node, sets value on node as attribute; returns graph, and dict of the eigenvector centrality values.
	Also has commented out code to sort by ec value
	'''
	g = graph
	ec = nx.eigenvector_centrality(g)
	nx.set_node_attributes(g,'eigen_cent',ec)
	#ec_sorted = sorted(ec.items(), key=itemgetter(1), reverse=True)
	return g, ec

def calculate_degree_centrality(graph):
	''' Calculate degree centrality of a node, sets value on node as attribute; returns graph, and dict of the degree centrality values.
	Also has code to print the top 10 nodes by degree centrality to console
	'''
	g = graph
	dc = nx.degree_centrality(g)
	nx.set_node_attributes(g,'degree_cent',dc)
	degcent_sorted = sorted(dc.items(), key=itemgetter(1), reverse=True)
	for key,value in degcent_sorted[0:10]:
		print "Highest degree Centrality:", key, value
	return graph, dc

def find_cliques(graph):
	''' Calculate cliques and return as sorted list.  Print sizes of cliques found.
	'''
	g = graph
	cl = nx.find_cliques(g)
	cl = sorted(list( cl ), key=len, reverse=True)
	print "Number of cliques:", len(cl)
	cl_sizes = [len(c) for c in cl]
	print "Size of cliques:", cl_sizes
	return cl
	
def find_partition(graph):
	''' Calculate partition membership, or subcommunities
	Requires code and lib from http://perso.crans.org/aynaud/communities/
	Requires an undirected graph - so convert it first.
	Returns graph, partition which is dict.  Updates the graph nodes with partition membership.
	Has commented out code that will report partition for each node.
	'''
	import community  		# download the code from link above and put in same dir.
	g = graph
	partition = community.best_partition( g )
	print "Partitions found: ", len(set(partition.values()))
	# Uncomment this to show members of each partition:
	#for i in set(partition.values()):
		#members = [nodes for nodes in partition.keys() if partition[nodes] == i]
		#for member in members:
			#print member, i
	#print "Partition for node Arnicas: ", partition["arnicas"]
	nx.set_node_attributes(g,'partition',partition)
	return g, partition
	 
def add_partitions_to_digraph(graph, partitiondict):
	''' Add the partition numbers to a graph - in this case, using this to update the digraph, with partitions calc'd off the undirected graph. Yes, it's a bad hack.
	'''
	g = graph
	nx.set_node_attributes(g, 'partition', partitiondict)
	nx.info(g)
	return
		

def numeric_comp(a,b):
	print a, b, float(b) > float(a)
	return float(b) > float(a)


edge_features_to_build = [common_friends_in,common_friends_out,common_friends_bi,jaccard,is_reciprocal,preferential_attachment,transitive_friends]
node_features_to_build = [in_degree,out_degree]

def build_features(edge_features_to_build,node_features_to_build,edge_list,G):
	training_set = dict()
	for feature in edge_features_to_build:
		training_set[feature.__name__] = list()
	#log the edge names
	training_set['edge'] = []
	for node in ['v1','v2']:
		for feature in node_features_to_build:
			training_set[feature.__name__+'_'+node] = list()
	# build all edges list for the friends_measure calculation
	for edge in edge_list:
		training_set['edge'].append(str(edge[0])+'_'+str(edge[1]))
		for feature in edge_features_to_build:
			training_set[feature.__name__].append(feature(G,edge[0],edge[1]))
		names = ['v1','v2']
		for i in range(len(names)):
			for feature in node_features_to_build:
				training_set[feature.__name__+'_'+names[i]].append(feature(G,edge[i]))
	return DataFrame(training_set)

