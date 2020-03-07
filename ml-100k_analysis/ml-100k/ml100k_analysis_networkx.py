import matplotlib
import matplotlib.pyplot as plt
import networkx as nx
import collections

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
    for i in range(len(users) - 1):
        for j in range(i + 1, len(users)):
            G.add_edge(users[i][0], users[j][0], rating=(users[i][1], users[j][1]), movie_name=movie_name_list[key][0],
                       movie_id=key, movie_genre = movie_name_list[key][1])

print("Number of nodes " + str(G.number_of_nodes()))
print("Number of edges " + str(G.number_of_edges()))
print("Network Density " + str(nx.density(G)))
print("Degree Associativity Coefficient " + str(nx.degree_assortativity_coefficient(G)))

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


calculate_degree_histogram('gender', ['M', 'F'])
calculate_clustering_coefficients('gender', ['M', 'F'])


plt.show()
