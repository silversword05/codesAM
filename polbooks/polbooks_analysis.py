import collections

import matplotlib
import matplotlib.pyplot as plt
import networkx as nx

matplotlib.use('TkAgg')
plt.style.use('seaborn-deep')

G = nx.read_gml('polbooks.gml')

node_dics = dict([('l', []), ('n', []), ('c', [])])

for c, node in enumerate(G.nodes(data=True)):
    node_dics[node[1]['value']].append(node[0])

print("Number of nodes " + str(G.number_of_nodes()))
print("Number of edges " + str(G.number_of_edges()))
print("Network Density " + str(nx.density(G)))
print("Degree Associativity Coefficient " + str(nx.degree_assortativity_coefficient(G)))

print("Number of liberal nodes " + str(len(node_dics['l'])))
print("Number of neutral nodes " + str(len(node_dics['n'])))
print("Number of conservative nodes " + str(len(node_dics['c'])))


def calculate_clustering_coefficients_and_degree(feat, feat_ids):
    node_wrt_feat_ids = dict([(x, []) for x in feat_ids])

    for node in G.nodes:
        if feat in G.nodes[node] and G.nodes[node][feat] in feat_ids:
            node_wrt_feat_ids[G.nodes[node][feat]].append(node)

    for c, x in enumerate(feat_ids):
        print("Number of nodes of feature " + feat + " of id " + str(x) + " is " + str(len(node_wrt_feat_ids[x])))

        nds = [p for p in node_wrt_feat_ids[x]]
        data = nx.clustering(G, nds)
        clustering_coefficients_count = collections.Counter([round(data[x], 4) for x in nds])

        print("CLUSTERING COEFFICIENT : " + feat + " " + str(x), end=' ')
        print(clustering_coefficients_count)
        clustering_coefficients, cnt = zip(*sorted(clustering_coefficients_count.most_common(), key=lambda x: x[0]))
        plt.figure(str(c) + 'clustering coefficients')
        plt.loglog(clustering_coefficients, cnt, color='b')
        plt.title("Clustering Coefficients " + feat + " " + str(x))
        plt.ylabel("Count")
        plt.xlabel("Clustering Coefficient")

        degree_sequence = sorted([d for n, d in G.degree(node_wrt_feat_ids[x])], reverse=True)
        degree_count = collections.Counter(degree_sequence)
        print("DEGREE : " + feat + " " + str(x), end=' ')
        print(degree_count)
        deg, cnt = zip(*sorted(degree_count.items(), key=lambda x: x[0]))
        plt.figure(str(c) + " degree")
        plt.loglog(deg, cnt, color='b')
        plt.title("Degree " + feat + " " + str(x))
        plt.ylabel("Count")
        plt.xlabel("Degree")

        associativity = nx.degree_assortativity_coefficient(G, nodes=nds)
        print("Associativity of feature " + feat + " of feature id " + str(x) + " is " + str(associativity))


calculate_clustering_coefficients_and_degree('value', ['l', 'n', 'c'])

plt.show()
