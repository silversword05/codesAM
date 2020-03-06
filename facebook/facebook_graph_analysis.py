import networkx as nx
import collections
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('TkAgg')
plt.style.use('seaborn-deep')

G = nx.Graph()

file = open("facebook/facebook_combined.txt")

for line in file.readlines():
    nodes = [int(x.strip(' ')) for x in line.strip('\n').split(' ')]
    G.add_edge(nodes[0], nodes[1])

file.close()

feat_name_file_list = ['facebook/0.featnames', 'facebook/107.featnames', 'facebook/348.featnames',
                       'facebook/414.featnames', 'facebook/698.featnames', 'facebook/1684.featnames',
                       'facebook/1912.featnames', 'facebook/3437.featnames', 'facebook/3980.featnames']
feat_file_list = ['facebook/0.feat', 'facebook/107.feat', 'facebook/348.feat', 'facebook/414.feat', 'facebook/698.feat',
                  'facebook/1684.feat', 'facebook/1912.feat', 'facebook/3437.feat', 'facebook/3980.feat']
ego_feat_file_list = ['facebook/0.egofeat', 'facebook/107.egofeat', 'facebook/348.egofeat', 'facebook/414.egofeat',
                      'facebook/698.egofeat', 'facebook/1684.egofeat', 'facebook/1912.egofeat', 'facebook/3437.egofeat',
                      'facebook/3980.egofeat']
ego_nds_id = [0, 107, 348, 414, 698, 1684, 1912, 3437, 3980]

feat_names = set()
feat_id_dics = {}

for i in range(len(feat_name_file_list)):
    feat_name_file = open(feat_name_file_list[i], 'r')
    feat_name_list = []
    for x in feat_name_file.readlines():
        lin = x.strip('\n').split(' ')
        feat_name = ' '.join(lin[1].split(';')[:-1])
        feat_names.add(feat_name)
        feat_name_list.append((feat_name, int(lin[-1].strip(' '))))
    feat_name_file.close()

    for x in feat_names:
        if x not in feat_id_dics:
            feat_id_dics.update({x: set()})

    for x in feat_name_list:
        feat_id_dics[x[0]].add(x[1])

    feat_file = open(feat_file_list[i], 'r')
    for x in feat_file.readlines():
        lin = x.strip('\n').split(' ')
        nid = int(lin[0])
        for p, val in enumerate(lin[1:]):
            if int(val) == 1:
                G.nodes[nid][feat_name_list[p][0]] = feat_name_list[p][1]
    feat_file.close()

    ego_feat_file = open(ego_feat_file_list[i])
    lin = ego_feat_file.readline().strip('\n').split(' ')
    for p, val in enumerate(lin):
        if val == 1:
            G.nodes[ego_nds_id[i]][feat_name_list[p][0]] = feat_name_list[p][1]

nx.write_graphml(G, 'facebook.graphml')

print("Anonymize Ids")
for x in feat_id_dics.keys():
    print(x + " ", end='')
    print(list(feat_id_dics[x]))

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
        clustering_coefficients, cnt = zip(*sorted(clustering_coefficients_count.most_common(), key=lambda x : x[0]))
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
        deg, cnt = zip(*sorted(degree_count.items(), key = lambda x : x[0] ))
        plt.figure(str(c) + " degree")
        plt.plot(deg, cnt, color='b')
        plt.title("Degree " + feat + " " + str(x))
        plt.ylabel("Count")
        plt.xlabel("Degree")


# Mention the feat name and the corresponding ids of the annonymized geature
calculate_degree_histogram('gender', [77, 78])
calculate_clustering_coefficients('gender', [77, 78])

plt.show()
