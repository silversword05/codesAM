import collections
import sys

import matplotlib
import matplotlib.pyplot as plt
import networkx as nx

matplotlib.use('TkAgg')
plt.style.use('seaborn-deep')

G = nx.Graph()

file = open('ml-1m/users.data', 'r')
for line in file.readlines():
    data = line.strip(' \n').split('::')
    G.add_node(int(data[0]), age=int(data[2]), gender=data[1], occupation=data[3], zip_code=data[4])
file.close()

file = open('ml-1m/movies.data', 'r')
movie_ids = [int(line.strip(' \n').split('::')[0]) for line in file.readlines()]
print("Total Number of Movies " + str(len(movie_ids)))
file.close()

movie_dias = dict([(x, []) for x in movie_ids])
file = open('ml-1m/ratings.data')
for line in file.readlines():
    data = line.strip(' \n').split('::')
    movie_id = int(data[1])
    movie_dias[movie_id].append((int(data[0]), int(data[2])))  # (user_id, rating)
file.close()

for key in movie_dias.keys():
    users = movie_dias[key]
    sys.stdout.write('\r' + "Performing movie key " + str(key))
    for i in range(len(users) - 1):
        for j in range(i + 1, len(users)):
            if not G.has_edge(users[i][0], users[j][0]):
                G.add_edge(users[i][0], users[j][0], rating=[(users[i][1], users[j][1])], movie_id=[key])
            else:
                G.edges[users[i][0], users[j][0]]['rating'].append((users[i][1], users[j][1]))
                G.edges[users[i][0], users[j][0]]['movie_id'].append(key)

print("Number of nodes " + str(G.number_of_nodes()))
print("Number of edges " + str(G.number_of_edges()))
print("Network Density " + str(nx.density(G)))
print("Degree Associativity Coefficient " + str(nx.degree_assortativity_coefficient(G)))


# nx.write_graphml(G, 'ml100k.gml')

def calculate_clustering_coefficients_and_degree(feat, feat_ids):
    node_wrt_feat_ids = dict([(x, []) for x in feat_ids])

    for node in G.nodes:
        if feat in G.nodes[node] and G.nodes[node][feat] in feat_ids:
            node_wrt_feat_ids[G.nodes[node][feat]].append(node)

    for c, x in enumerate(feat_ids):
        print("Number of nodes of feature " + feat + " of id " + str(x))

        nds = [p for p in node_wrt_feat_ids[x]]
        data = nx.clustering(G, nds)
        clustering_coefficients_count = collections.Counter([round(data[x], 4) for x in nds])

        print("CLUSTERING COEFFICIENT : " + feat + " " + str(x), end=' ')
        print(clustering_coefficients_count)
        clustering_coefficients, cnt = zip(*sorted(clustering_coefficients_count.most_common(), key=lambda x: x[0]))
        plt.figure(str(c) + 'clustering coefficients')
        plt.plot(clustering_coefficients, cnt, color='b')
        plt.title("Clustering Coefficients " + feat + " " + str(x))
        plt.ylabel("Count")
        plt.xlabel("Clustering Coefficient")

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

        associativity = nx.degree_assortativity_coefficient(G, nodes=nds)
        print("Associativity of feature " + feat + " of feature id " + str(x) + " is " + str(associativity))


calculate_clustering_coefficients_and_degree('gender', ['M', 'F'])

plt.show()
