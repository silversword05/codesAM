import random as rand

import networkx as nx
from gensim.models import Word2Vec
from node2vec import Node2Vec

G = nx.read_graphml('facebook.graphml')  # only gender attributes


def precomputed_neighbour_features(features: list, feature_key: str) -> dict:
    alias_features = {}
    for node in G.nodes:
        feature_dic = [(x, []) for x in features]
        feature_dic = dict(feature_dic)
        for neighbour in G.neighbors(node):
            try:
                feature = G.nodes[neighbour][feature_key]
                feature_dic[feature].append(neighbour)
            except:
                print(feature_key + " not found in nodeid " + str(neighbour) + " setting to " + str(features[0]))
                G.nodes[neighbour][feature_key] = features[0]
        alias_features.update({node: feature_dic})

    return alias_features


def wash_dict(features_dic: dict) -> dict:
    keys = features_dic.keys()
    for key in list(keys):
        if len(features_dic[key]) == 0:
            del features_dic[key]
    return features_dic


def fair_walk(walk_num: int, walk_len: int, alias_features: dict) -> list:
    walks = []
    for node in G.nodes:
        for i in range(0, walk_num):
            walk = []
            curr = node
            for j in range(0, walk_len):
                walk.append(curr)
                features_dic = wash_dict(alias_features[curr])
                random_feature = rand.choice(list(features_dic.keys()))
                curr = rand.choice(features_dic[random_feature])
            walks.append(walk)
    return walks


alis_features_dic = precomputed_neighbour_features([77, 78], 'gender')
print()
walks_word2vec = fair_walk(walk_num=20, walk_len=80, alias_features=alis_features_dic)
print("Total number of walks " + str(len(walks_word2vec)))

model = Word2Vec(walks_word2vec, size=128, window=10, min_count=0, sg=1, workers=8, iter=1)
model.wv.save_word2vec_format('facebook_fair_walk.emb')

node2vec = Node2Vec(G, dimensions=128, walk_length=80, num_walks=20, workers=8)
model = node2vec.fit(window=10, min_count=0, sg=1, iter=1)
model.wv.save_word2vec_format('facebook_rand_walk.emb')
