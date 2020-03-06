import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import snap

matplotlib.use('TkAgg')
plt.style.use('seaborn-deep')

RATING_NOS = 5

graph = snap.TNEANet.New()

# Adding Occupations
f = open('ml-100k/ml-100k/u.occupation')
occupation_dics = {}
for x in f.readlines():
    nid = graph.AddNode(-1)
    graph.AddStrAttrDatN(nid, x.strip(' \n'), "Occupation")
    occupation_dics.update({x.strip(' \n'): nid})

print(occupation_dics)
f.close()

# Adding Genders
gender_dics = {}
for x in ['M', 'F']:
    nid = graph.AddNode(-1)
    graph.AddStrAttrDatN(nid, x, "Gender")
    gender_dics.update({x: nid})

print(gender_dics)

# Adding Users
f = open('ml-100k/ml-100k/u.user')
user_dics = {}
for x in f.readlines():
    nid = graph.AddNode(-1)
    data = x.strip(' \n').split('|')
    graph.AddIntAttrDatN(nid, int(data[0]), "Id")
    graph.AddIntAttrDatN(nid, int(data[1]), "Age")
    graph.AddEdge(gender_dics[data[2]], nid, -1)
    graph.AddEdge(occupation_dics[data[3]], nid, -1)
    # if data[2] == 'F' and data[3] == 'doctor':
    #     print("Wrong")
    graph.AddStrAttrDatN(nid, data[4], "zip")
    user_dics.update({"Id-" + data[0]: nid})

print(user_dics)
f.close()

# Adding Genre
f = open('ml-100k/ml-100k/u.genre')
genre_dics = {}
genre_list = []
for x in f.readlines():
    data = x.strip(' \n').split('|')
    nid = graph.AddNode(-1)
    graph.AddStrAttrDatN(nid, data[0], "Genre")
    graph.AddIntAttrDatN(nid, int(data[1].strip(' \n')), "Id")
    genre_dics.update({data[0]: nid})
    genre_list.append(data[0])

print(genre_dics)
f.close()

# Adding Films
f = open('ml-100k/ml-100k/u.item')
item_dics = {}
for x in f.readlines():
    nid = graph.AddNode(-1)
    data = x.strip(' \n').split('|')
    graph.AddIntAttrDatN(nid, int(data[0]), "Id")
    graph.AddStrAttrDatN(nid, data[1], "Title")
    graph.AddStrAttrDatN(nid, data[2], "Release_Date")
    graph.AddStrAttrDatN(nid, data[3], "Video_Release_Date")
    graph.AddStrAttrDatN(nid, data[4], "IMDB URL")
    genres = [int(x) for x in data[5:]]
    for i in range(len(genres)):
        if genres[i] == 1:
            graph.AddEdge(genre_dics[genre_list[i]], nid, -1)
    item_dics.update({"Id-" + data[0]: nid})

print(item_dics)
f.close()

# Adding Ratings
f = open('ml-100k/ml-100k/u.data')
for x in f.readlines():
    data = x.strip(' \n').split('\t')
    item_nid = item_dics["Id-" + data[1]]
    user_nid = user_dics["Id-" + data[0]]
    eid = graph.AddEdge(user_nid, item_nid, -1)
    graph.AddIntAttrDatE(eid, int(data[2]), "Rating")
    graph.AddStrAttrDatE(eid, data[3], "Timestamp")

print("Successfully created graph")
print()

f.close()

FOut = snap.TFOut("test.graph")
graph.Save(FOut)
FOut.Flush()
# snap.DrawGViz(graph, snap.gvlDot, "graph.png", "graph 1")

# get female node iterator
female_node = graph.GetNI(gender_dics['F'])
# get male node iterator
male_node = graph.GetNI(gender_dics['M'])


def findRatingMatrixOcc(gender_node):
    gae_ratings = {}
    for key in occupation_dics.keys():
        gae_ratings.update({key: []})
    for x in gender_node.GetOutEdges():
        # finding users who are that gender
        user = graph.GetNI(x)
        for y in user.GetOutEdges():
            # finding the edge contains the rating information
            edge_iterator = graph.GetEI(user.GetId(), y)
            for occ, occ_id in occupation_dics.items():
                # checking the occupation of the user
                if graph.IsEdge(occ_id, user.GetId()):
                    gae_ratings[occ].append(graph.GetIntAttrDatE(edge_iterator, "Rating"))

    rating_matrix = np.zeros(shape=[len(gae_ratings.keys()), RATING_NOS], dtype=int)
    for key, value in gae_ratings.items():
        val_arr = np.array(value)
        for i in range(1, RATING_NOS + 1):
            rating_matrix[list(gae_ratings.keys()).index(key)][i - 1] = int(np.count_nonzero(val_arr == i))

    rating_matrix = [[y * 100 / sum(x) for y in x] for x in rating_matrix]
    print(rating_matrix)
    return rating_matrix


def findRatingMatrixGenre(gender_node):
    gae_ratings = {}
    for key in genre_dics.keys():
        gae_ratings.update({key: []})
    for x in gender_node.GetOutEdges():
        # finding users who are that gender
        user = graph.GetNI(x)
        for y in user.GetOutEdges():
            # finding the edge contains the rating information
            edge_iterator = graph.GetEI(user.GetId(), y)
            for genre, genre_id in genre_dics.items():
                # checking the occupation of the user
                if graph.IsEdge(genre_id, y):
                    gae_ratings[genre].append(graph.GetIntAttrDatE(edge_iterator, "Rating"))

    rating_matrix = np.zeros(shape=[len(gae_ratings.keys()), RATING_NOS], dtype=int)
    for key, value in gae_ratings.items():
        val_arr = np.array(value)
        for i in range(1, RATING_NOS + 1):
            rating_matrix[list(gae_ratings.keys()).index(key)][i - 1] = int(np.count_nonzero(val_arr == i))

    rating_matrix = [[y * 100 / sum(x) for y in x] for x in rating_matrix]
    print(rating_matrix)
    return rating_matrix


print("Occupation")
print("Male ", end=' ')
ratingMatrixMale = findRatingMatrixOcc(male_node)
print("Female ", end=' ')
ratingMatrixFemale = findRatingMatrixOcc(female_node)

# plt.figure(0)
fig, axs = plt.subplots(3, 7)
fig.suptitle('Occupation : Blue for male and Green for female', fontsize=12)

for i in range(3):
    for j in range(7):
        axs[i, j].plot(list(range(1, RATING_NOS + 1)), ratingMatrixMale[i * 7 + j], linestyle='--', marker='o',
                       color='b')
        axs[i, j].plot(list(range(1, RATING_NOS + 1)), ratingMatrixFemale[i * 7 + j], linestyle='--', marker='o',
                       color='g')
        axs[i, j].set_xticks(list(range(1, RATING_NOS + 1)))
        # axs[i, j].set_ylabel("% of population")
        axs[i, j].set_title(" Occupation " + list(occupation_dics.keys())[i * 7 + j])

plt.figure(1)
print("Genre")
print("Male ", end=' ')
ratingMatrixMale = findRatingMatrixGenre(male_node)
print("Female ", end=' ')
ratingMatrixFemale = findRatingMatrixGenre(female_node)

fig, axs = plt.subplots(3, 7)
fig.suptitle('Genre : Blue for male and Green for female', fontsize=12)

for i in range(3):
    for j in range(7):
        if i * 7 + j == 19:
            break
        axs[i, j].plot(list(range(1, RATING_NOS + 1)), ratingMatrixMale[i * 7 + j], linestyle='--', marker='o',
                       color='b')
        axs[i, j].plot(list(range(1, RATING_NOS + 1)), ratingMatrixFemale[i * 7 + j], linestyle='--', marker='o',
                       color='g')
        axs[i, j].set_xticks(list(range(1, RATING_NOS + 1)))
        axs[i, j].set_title(" Genre " + list(genre_dics.keys())[i * 7 + j])

plt.show()
