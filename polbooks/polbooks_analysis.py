import networkx as nx
import collections
import matplotlib.pyplot as plt
import matplotlib

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


def calculate_clustering_coefficients(feat, feat_ids):
    node_wrt_feat_ids = dict([(x, []) for x in feat_ids])

    for node in G.nodes:
        if feat in G.nodes[node] and G.nodes[node][feat] in feat_ids:
            node_wrt_feat_ids[G.nodes[node][feat]].append(node)

    for c, x in enumerate(feat_ids):
        nds = [p for p, y in G.nodes(data=True) if feat in y and y[feat] == x]
        data = nx.clustering(G, nds)
        clustering_coefficients_count = collections.Counter([data[x] for x in nds])

        print("CLUSTERING COEFFICIENT : " + feat + " " + str(x), end=' ')
        print(clustering_coefficients_count)
        clustering_coefficients, cnt = zip(*sorted(clustering_coefficients_count.most_common(), key=lambda x: x[0]))
        plt.figure(str(c) + 'clustering coefficients')
        plt.plot(clustering_coefficients, cnt, color='b')
        plt.title("Clustering Coefficients " + feat + " " + str(x))
        plt.ylabel("Count")
        plt.xlabel("Clustering Coefficient")


def calculate_degree_histogram(feat, feat_ids):
    node_wrt_feat_ids = dict([(x, []) for x in feat_ids])

    for node in G.nodes:
        if feat in G.nodes[node] and G.nodes[node][feat] in feat_ids:
            node_wrt_feat_ids[G.nodes[node][feat]].append(node)

    for c, x in enumerate(feat_ids):
        degree_sequence = sorted([d for n, d in G.degree(node_wrt_feat_ids[x])], reverse=True)
        degree_count = collections.Counter(degree_sequence)
        print("DEGREE : " + feat + " " + str(x), end=' ')
        print(degree_count)
        deg, cnt = zip(*sorted(degree_count.items(), key=lambda x: x[0]))
        plt.figure(str(c) + " degree")
        plt.plot(deg, cnt, color='b')
        plt.title("Degree " + feat + " " + str(x))
        plt.ylabel("Count")
        plt.xlabel("Degree")


calculate_degree_histogram('value', ['l', 'n', 'c'])
calculate_clustering_coefficients('value', ['l', 'n', 'c'])

plt.show()
