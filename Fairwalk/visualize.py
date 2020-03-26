import numpy as np
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE


def get_coordinates(filename):
    labels = []
    f = open(filename, 'r')
    lines = f.readlines()
    word_count, dimension = [int(x.strip(' ')) for x in lines[0].strip(' \n').split(' ')]
    arr = np.empty((0, dimension), dtype='f')
    for line in lines[1:]:
        data = line.strip(' \n').split(' ')
        labels.append(data[0])
        data = [float(x) for x in data[1:]]
        arr = np.append(arr, np.array(data).reshape(1, len(data)), axis=0)

    print(len(arr))
    tsne = TSNE(n_components=2, random_state=0)
    np.set_printoptions(suppress=True)
    Y = tsne.fit_transform(arr)
    x_coords = Y[:, 0]
    y_coords = Y[:, 1]
    return x_coords, y_coords, labels


x_fair, y_fair, labels_fair = get_coordinates('facebook_fair_walk.emb')
x_rand, y_rand, labels_rand = get_coordinates('facebook_rand_walk.emb')

plt.figure(figsize=(14, 8))
plt.scatter(x_fair, y_fair, c='steelblue', edgecolors='k')
plt.scatter(x_rand, y_rand, c='darkblue', edgecolors='k')
# for label, x, y in zip(labels, x, y):
#     plt.annotate(label, xy=(x+1, y+1), xytext=(0, 0), textcoords='offset points')

plt.show()
