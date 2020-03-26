import networkx as nx
import numpy as np
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

G: nx.Graph = nx.read_edgelist('facebook.edgelist')

filename = 'facebook_fair_walk.emb'
f = open(filename, 'r')
lines = f.readlines()
word_count, dimension = [int(x.strip(' ')) for x in lines[0].strip(' \n').split(' ')]
node_ids = []
node_embeddings = {}
for line in lines[1:]:
    data = line.strip(' \n').split(' ')
    node_ids.append(data[0])
    data = [float(x) for x in data[1:]]
    node_embeddings.update({node_ids[-1]: np.array(data).reshape(1, len(data))})

print("Reading embeddings complete")
print("Total number of nodes " + str(len(node_ids)))
node_ids = np.array(node_ids)

input_data = np.empty((0, dimension), dtype='f')
input_labels = np.zeros(0, dtype='i')
counter = 0
for node_id in node_ids:
    neighbours = list(G.neighbors(node_id))
    print("\rProcessing for node id " + str(node_id) + " number of neighbours " + str(
        len(neighbours)) + " of position " + str(counter), end='')
    counter += 1
    random_nodes = node_ids[
        np.random.choice(len(node_ids), min(int(len(neighbours) * 1.5), len(node_ids)), replace=False)]
    random_nodes_filtered = [x for x in random_nodes if x not in neighbours][:len(neighbours)]
    for x in random_nodes_filtered:
        input_data = np.append(input_data, np.multiply(node_embeddings[node_id], node_embeddings[x]), axis=0)
        input_labels = np.append(input_labels, 0)
    for x in neighbours:
        input_data = np.append(input_data, np.multiply(node_embeddings[node_id], node_embeddings[x]), axis=0)
        input_labels = np.append(input_labels, 1)

print(len(input_labels))
print(len(input_data))

randomize_data_points = np.arange(len(input_labels))
np.random.shuffle(randomize_data_points)
input_labels = input_labels[randomize_data_points]
input_data = input_labels[randomize_data_points]

clf = RandomForestClassifier(n_estimators=100)
X_train, X_test, y_train, y_test = train_test_split(input_data, input_labels, test_size=0.3)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
