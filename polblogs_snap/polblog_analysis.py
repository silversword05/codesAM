import matplotlib
import matplotlib.pyplot as plt
import snap

matplotlib.use('TkAgg')
plt.style.use('seaborn-deep')

graph = snap.TNEANet.New()

file = open('polblogs.gml')

while True:
    line = file.readline().strip('\n ')
    if len(line) == 0:
        break

    if 'node' in line:
        idz = int(file.readline().replace('id', '').strip(' \n'))
        label = file.readline().replace('label', '').strip(' \n')
        value = int(file.readline().replace('value', '').strip(' \n'))
        value = 'liberal' if value == 0 else 'conservative'
        source = file.readline().replace('source', '').replace('"', '').strip(' \n')
        nid = graph.AddNode(idz)
        graph.AddIntAttrDatN(nid, idz, "id")
        graph.AddStrAttrDatN(nid, value, "value")
        graph.AddStrAttrDatN(nid, source, "source")

    if 'edge' in line:
        src = int(file.readline().replace('source', '').strip(' \n'))
        target = int(file.readline().replace('target', '').strip(' \n'))
        graph.AddEdge(src, target)

file.close()

FOut = snap.TFOut("polblogs.graph")
graph.Save(FOut)
FOut.Flush()

print("Number of nodes " + str(graph.GetNodes()))
print("Number of edges " + str(graph.GetEdges()))

avg_in_degree, avg_out_degree = 0, 0
deg_nds = []
no_liberal, no_conservative = 0, 0

liberal_deg_nds, conservative_deg_nds = [], []

for node in graph.Nodes():
    idx = graph.GetIntAttrDatN(node, "id")
    in_deg = node.GetInDeg()
    out_deg = node.GetOutDeg()
    deg_nds.append(node.GetDeg())
    avg_in_degree += in_deg
    avg_out_degree += out_deg

    value = graph.GetStrAttrDatN(node, "value")
    if value == 'liberal':
        no_liberal += 1
        liberal_deg_nds.append(node.GetDeg())
    else:
        no_conservative += 1
        conservative_deg_nds.append(node.GetDeg())

nde_cnt_freq = {}
nde_cnt_freq_liberal = {}
nde_cnt_freq_conservative = {}

for x in set(deg_nds):
    freq = deg_nds.count(x)
    nde_cnt_freq.update({x: freq})

for x in set(liberal_deg_nds):
    freq = liberal_deg_nds.count(x)
    nde_cnt_freq_liberal.update({x: freq})

for x in set(conservative_deg_nds):
    freq = conservative_deg_nds.count(x)
    nde_cnt_freq_conservative.update({x: freq})

print("Average in Degree " + str(avg_in_degree / graph.GetNodes()))
print("Average out Degree " + str(avg_out_degree / graph.GetNodes()))

print("Liberal Nodes " + str(no_liberal))
print("Conservative Nodes " + str(no_conservative))

print("Density of graph " + str(graph.GetEdges() / (graph.GetNodes() * (graph.GetNodes() - 1))))

snap.PlotClustCf(graph , "cluster_plot", "Polblog Directed graph - clustering coefficient")

fig, axs = plt.subplots(3)

x_val = list(nde_cnt_freq_conservative.keys())
y_val = list(nde_cnt_freq_conservative.values())
axs[0].plot(x_val, y_val)
axs[0].set_ylabel("Frequency of that degree ")
axs[0].set_title("Degree of the conservative nodes")

x_val = list(nde_cnt_freq_liberal.keys())
y_val = list(nde_cnt_freq_liberal.values())
axs[1].plot(x_val, y_val)
axs[1].set_ylabel("Frequency of that degree ")
axs[1].set_title("Degree of the liberal nodes")

x_val = list(nde_cnt_freq.keys())
y_val = list(nde_cnt_freq.values())
axs[2].plot(x_val, y_val)
axs[2].set_ylabel("Frequency of that degree ")
axs[2].set_title("Degree of the entire nodes")

plt.show()
