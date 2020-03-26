import collections
import sys

import matplotlib
import matplotlib.pyplot as plt
import networkx as nx

matplotlib.use('TkAgg')
plt.style.use('seaborn-deep')

G = nx.Graph()

file = open('ml-100k/u.item', 'r')
movie_name_list = {}
for line in file.readlines():
    name = line.strip(' \n').split('|')[1]
    idz = int(line.strip(' \n').split('|')[0])
    genre_list = [int(x) for x in line.strip(' \n').split('|')[5:]]
    movie_name_list.update({idz: (name, genre_list)})
file.close()

file = open('ml-100k/u.data')
movie_dias = dict([(x, []) for x in movie_name_list.keys()])
for line in file.readlines():
    data = line.strip(' \n').split('\t')
    item_id = int(data[1])
    movie_dias[item_id].append((int(data[0]), int(data[2])))  # (user_id, rating)
file.close()

file = open('ml-100k/u.user')
for line in file.readlines():
    data = line.strip(' \n').split('|')
    G.add_node(int(data[0]), age=int(data[1]), gender=data[2], occupation=data[3], zip_code=data[4])
file.close()

for key in movie_dias.keys():
    users = movie_dias[key]
    sys.stdout.write('\r' + "Performing movie key " + str(key))
    for i in range(len(users) - 1):
        for j in range(i + 1, len(users)):
            if not G.has_edge(users[i][0], users[j][0]):
                G.add_edge(users[i][0], users[j][0], rating=[(users[i][1], users[j][1])],
                           movie_name=[movie_name_list[key][0]], movie_id=[key], movie_genre=[movie_name_list[key][1]])
            else:
                G.edges[users[i][0], users[j][0]]['rating'].append((users[i][1], users[j][1]))
                G.edges[users[i][0], users[j][0]]['movie_name'].append(movie_name_list[key][0])
                G.edges[users[i][0], users[j][0]]['movie_id'].append(key)
                G.edges[users[i][0], users[j][0]]['movie_genre'].append(movie_name_list[key][1])

print()
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


calculate_clustering_coefficients_and_degree('gender', ['M', 'F'])

# node2vec = Node2Vec(G, dimensions=64, walk_length=30, num_walks=200, workers=1)
# model = node2vec.fit(window=10, min_count=1, batch_words=4)
# player_nodes = [x for x in model.wv.vocab if len(x) > 3]
# embeddings = np.array([model.wv[x] for x in player_nodes])
# tsne = TSNE(n_components=2, random_state=7, perplexity=15)
# embeddings_2d = tsne.fit_transform(embeddings)
# figure = plt.figure(figsize=(11, 9))
# ax = figure.add_subplot(111)
# ax.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1])

plt.show()
