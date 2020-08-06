import multiprocessing

import networkx as nx
import numpy as np
from joblib import Parallel, delayed
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

G: nx.Graph = nx.read_edgelist('facebook.edgelist')
node_ids: list = []
node_embeddings = {}
results_input = []
num_cores = multiprocessing.cpu_count()


def read_data():
    global node_embeddings, node_ids
    filename = 'facebook_fair_walk.emb'
    f = open(filename, 'r')
    lines = f.readlines()
    word_count, dimension = [int(x.strip(' ')) for x in lines[0].strip(' \n').split(' ')]
    print(f" Word Count : {word_count}  Dimension: {dimension}")
    for line in lines[1:]:
        data = line.strip(' \n').split(' ')
        node_ids.append(data[0])
        data = [float(x) for x in data[1:]]
        node_embeddings.update({node_ids[-1]: np.array(data).reshape(len(data))})

    print("Reading embeddings complete")
    print("Total number of nodes " + str(len(node_ids)))


def get_input_for_node(node_id):
    global G, node_ids, node_embeddings
    neighbours = list(G.neighbors(node_id))
    print("\rProcessing for node id " + str(node_id) + " number of neighbours " + str(len(neighbours)), end='')
    positions = np.random.choice(len(node_ids), min(int(len(neighbours) * 1.5), len(node_ids)), replace=False)
    random_nodes = [node_ids[x] for x in positions]
    random_nodes_filtered = [x for x in random_nodes if x not in neighbours][:len(neighbours)]
    input_node_id = []
    for x in random_nodes_filtered:
        input_node_id.append((np.multiply(node_embeddings[node_id], node_embeddings[x]), 0))
    for x in neighbours:
        input_node_id.append((np.multiply(node_embeddings[node_id], node_embeddings[x]), 1))
    return input_node_id


def execute_parallel_inputs():
    global results_input
    print("Starting Execution")
    node_ids.sort(key=lambda x: int(x))
    output = Parallel(n_jobs=num_cores)(delayed(get_input_for_node)(node_id) for node_id in node_ids)
    for i, out in enumerate(output):
        print(f"Total number of data points for node {node_ids[i]} is {len(out)}")
        results_input.extend(out)
    print("Total number of data points", len(results_input))
    print("Execution complete")


def execute_random_forest():
    global results_input
    input_data = np.array([elem[0] for elem in results_input])
    input_labels = np.array([elem[1] for elem in results_input])

    randomize_data_points = np.arange(len(input_labels))
    np.random.shuffle(randomize_data_points)
    input_labels = input_labels[randomize_data_points]
    input_data = input_data[randomize_data_points]

    clf = RandomForestClassifier(n_estimators=100)
    X_train, X_test, y_train, y_test = train_test_split(input_data, input_labels, test_size=0.3)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print("Accuracy:", metrics.accuracy_score(y_test, y_pred))


if __name__ == '__main__':
    read_data()
    execute_parallel_inputs()
    execute_random_forest()
