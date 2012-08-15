from networkx_functs import *
from utilities import *
from bfs_benchmark import breadth_first_search
import cPickle as pickle
import sys
import csv

def score_all(G,v1,v2):
	return np.sum([score1(friends_measure,G,v1,v2)+score2(friends_measure,G,v1,v2)+score3(friends_measure,G,v1,v2)])


if __name__ == '__main__':

	#with open('G.pkl') as f:
	#	G = pickle.load(f)
	M = build_sparse_matrix('train.csv')
	G = nx.from_scipy_sparse_matrix(M,create_using=nx.DiGraph())

	r = csv.reader(open('train.csv','r'))
	r.next()

	edges = set()
	for edge in r:
		edges.add((edge[0], edge[1]))
	
	missing_edges = set()
	for edge in edges:
		if (edge[1], edge[0]) not in edges:
			missing_edges.add((edge[1], edge[0]))

	missing_graph = dict()
	for edge in missing_edges:
		missing_graph.setdefault(edge[0], set()).add(edge[1])

	r = csv.reader(open('test.csv','r'))
	r.next()

	test_list = list()
	for line in r:
		test_list.append(line[0])

	test_lists = dict()
	for node in test_list:
		test_lists[node] = list(missing_graph.get(node, set()))


	results = list()
	for node in test_list[:10]:
		nodes_to_recommend = [int(i) for i in test_lists[node]]
		try:
			recs = rank_commutative_by_pagerank(G,nodes_to_recommend)
		except:
			recs = []
		results.append(recs)


	test_nodes = read_nodes_list('test.csv')
	write_submission_file('improve_commutative_pagerank.csv', test_nodes, results)

