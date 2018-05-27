import itertools
import copy
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
import imageio

edgelist = pd.read_csv('https://gist.githubusercontent.com/brooksandrew/e570c38bcc72a8d102422f2af836513b/'
                       'raw/89c76b2563dbc0e88384719a35cba0dfc04cd522/edgelist_sleeping_giant.csv')
# edgelist.head(10)

nodelist = pd.read_csv('https://gist.githubusercontent.com/brooksandrew/'
                       'f989e10af17fb4c85b11409fea47895b/raw/a3a8da0fa5b094f1ca9d82e1642b384889ae16e8/'
                       'nodelist_sleeping_giant.csv')

# Preview nodelist
# nodelist.head(5)

g = nx.Graph()
l=[]

for i, elrow in edgelist.iterrows():

    l.append(elrow[2:].to_dict())
    g.add_edge(elrow[0], elrow[1], distance = elrow['distance'],
               trail = elrow['trail'], color = elrow['color'], estimate = elrow['estimate'])          ################


for i, nlrow in nodelist.iterrows():
    g.node[nlrow['id']].update( nlrow[1:].to_dict() )



print('# of edges: {}'.format(g.number_of_edges()))
print('# of nodes: {}'.format(g.number_of_nodes()))

# print(g.nodes.columns)
node_positions = {node[0]: (node[1]['X'], -1*node[1]['Y']) for node in g.nodes(data=True)}
# print(dict(list(node_positions.items())[0:5]))

edge_colors = [e[2]['color'] for e in g.edges(data=True)]
plt.figure(figsize=(8, 6))
nx.draw(g, pos=node_positions, edge_color=edge_colors, node_size=10, node_color='black')
plt.title('Graph Representation of Sleeping Giant Trail Map', size=15)
plt.show()

# Calculate list of nodes with odd degree
nodes_odd_degree = [v for (v, d) in list(g.degree()) if d%2==1]
print('Number of nodes of odd degree: {}'.format(len(nodes_odd_degree)))
print('Number of total nodes: {}'.format(len(g.nodes())))

odd_node_pairs = list(itertools.combinations(nodes_odd_degree, 2))
# Counts
print('Number of pairs: {}'.format(len(odd_node_pairs)))

def get_shortest_paths_distances(graph, pairs, edge_weight_name):
    """Compute shortest distance between each pair of nodes in a graph.  Return a dictionary keyed on node pairs (tuples)."""
    distances = {}
    for pair in pairs:
        distances[pair] = nx.dijkstra_path_length(graph, pair[0], pair[1], weight=edge_weight_name)
    return distances

odd_node_pairs_shortest_paths = get_shortest_paths_distances(g, odd_node_pairs, 'distance')

def create_complete_graph(pair_weights, flip_weights=True):
    g = nx.Graph()
    for k, v in pair_weights.items():
        wt_i = - v if flip_weights else v
        g.add_edge( k[0], k[1], attr_dict={'distance': v, 'weight': wt_i} )
    return g

# Generate the complete graph
g_odd_complete = create_complete_graph(odd_node_pairs_shortest_paths, flip_weights=True)
# Counts
print('Number of nodes: {}'.format(len(g_odd_complete.nodes())))
print('Number of edges: {}'.format(len(g_odd_complete.edges())))

# Plot the complete graph of odd-degree nodes
plt.figure(figsize=(8, 6))
pos_random = nx.random_layout(g_odd_complete)
nx.draw_networkx_nodes(g_odd_complete, node_positions, node_size=20, node_color="red")
nx.draw_networkx_edges(g_odd_complete, node_positions, alpha=0.1)
plt.axis('off')
plt.title('Complete Graph of Odd-degree Nodes')
plt.show()

odd_matching_dupes = nx.algorithms.max_weight_matching(g_odd_complete, True)

print('Number of edges in matching: {}'.format(len(odd_matching_dupes)))

odd_matching = list(pd.unique([tuple(sorted([k, v])) for k, v in odd_matching_dupes.items()]))
print('Number of edges in matching (deduped): {}'.format(len(odd_matching)))

# for (i,j) in odd_matching:
#     print("("+i+ ", "+j+ ")")

plt.figure(figsize=(8, 6))
nx.draw(g_odd_complete, pos=node_positions, node_size=20, alpha=0.05)
g_odd_complete_min_edges = nx.Graph(odd_matching)
nx.draw(g_odd_complete_min_edges, pos=node_positions, node_size=20, edge_color='blue', node_color='red')

plt.title('Min Weight Matching on Complete Graph')
plt.show()

plt.figure(figsize=(8, 6))
nx.draw(g, pos=node_positions, node_size=20, alpha=0.1, node_color='black')
nx.draw(g_odd_complete_min_edges, pos=node_positions, node_size=20, alpha=1, node_color='red', edge_color='blue')
plt.title('Min Weight Matching on Orginal Graph')
plt.show()

def add_augmenting_path_to_graph(graph, min_weight_pairs):
    graph_aug = nx.MultiGraph(graph.copy())
    for pair in min_weight_pairs:
        graph_aug.add_edge(pair[0],
                           pair[1], distance = nx.dijkstra_path_length(graph, pair[0], pair[1]), trail = 'augmented'
                           # attr_dict={'distance': nx.dijkstra_path_length(graph, pair[0], pair[1]),
                           #            'trail': 'augmented'}
                          )
    return graph_aug


# Create augmented graph: add the min weight matching edges to g
g_aug = add_augmenting_path_to_graph(g, odd_matching)

# Counts
print('Number of edges in original graph: {}'.format(len(g.edges())))
print('Number of edges in augmented graph: {}'.format(len(g_aug.edges())))
l = []
for n in g_aug.nodes():
    l.append(g.degree(n))

naive_euler_circuit = list(nx.eulerian_circuit(g_aug, source='b_end_east'))
print('Length of eulerian circuit: {}'.format(len(naive_euler_circuit)))

def create_eulerian_circuit(graph_augmented, graph_original, starting_node=None):
    """Create the eulerian path using only edges from the original graph."""
    euler_circuit = []
    naive_circuit = list(nx.eulerian_circuit(graph_augmented, source=starting_node))

    for edge in naive_circuit:
        edge_data = graph_augmented.get_edge_data(edge[0], edge[1])

        if edge_data[0]['trail'] != 'augmented':
            # If `edge` exists in original graph, grab the edge attributes and add to eulerian circuit.
            edge_att = graph_original[edge[0]][edge[1]]
            euler_circuit.append((edge[0], edge[1], edge_att))
        else:
            aug_path = nx.shortest_path(graph_original, edge[0], edge[1], weight='distance')
            aug_path_pairs = list(zip(aug_path[:-1], aug_path[1:]))

            print('Filling in edges for augmented edge: {}'.format(edge))
            print('Augmenting path: {}'.format(' => '.join(aug_path)))
            print('Augmenting path pairs: {}\n'.format(aug_path_pairs))

            # If `edge` does not exist in original graph, find the shortest path between its nodes and
            #  add the edge attributes for each link in the shortest path.
            for edge_aug in aug_path_pairs:
                edge_aug_att = graph_original[edge_aug[0]][edge_aug[1]]
                euler_circuit.append((edge_aug[0], edge_aug[1], edge_aug_att))

    return euler_circuit

euler_circuit = create_eulerian_circuit(g_aug, g, 'b_end_east')
print('Length of Eulerian circuit: {}'.format(len(euler_circuit)))

def create_cpp_edgelist(euler_circuit):
    cpp_edgelist = {}

    for i, e in enumerate( euler_circuit ):
        edge = frozenset( [e[0], e[1]] )

        if edge in cpp_edgelist:
            cpp_edgelist[edge][2]['sequence'] += ', ' + str( i )
            cpp_edgelist[edge][2]['visits'] += 1

        else:
            cpp_edgelist[edge] = e
            cpp_edgelist[edge][2]['sequence'] = str( i )
            cpp_edgelist[edge][2]['visits'] = 1

    return list( cpp_edgelist.values() )

cpp_edgelist = create_cpp_edgelist(euler_circuit)
print('Number of edges in CPP edge list: {}'.format(len(cpp_edgelist)))


g_cpp = nx.Graph(cpp_edgelist)

#Visual:

visit_colors = {1:'black', 2:'red'}
edge_cnter = {}
g_i_edge_colors = []
filenames = []
for i, e in enumerate(euler_circuit, start=1):

    edge = frozenset([e[0], e[1]])
    if edge in edge_cnter:
        edge_cnter[edge] += 1
    else:
        edge_cnter[edge] = 1

    # Full graph (faded in background)
    nx.draw_networkx(g_cpp, pos=node_positions, node_size=6, node_color='gray', with_labels=False, alpha=0.07)

    # Edges walked as of iteration i
    euler_circuit_i = copy.deepcopy(euler_circuit[0:i])
    for i in range(len(euler_circuit_i)):
        edge_i = frozenset([euler_circuit_i[i][0], euler_circuit_i[i][1]])
        euler_circuit_i[i][2]['visits_i'] = edge_cnter[edge_i]
    g_i = nx.Graph(euler_circuit_i)
    g_i_edge_colors = []
    for e in g_i.edges( data=True ):
        if e[2]['visits_i']>1:
            g_i_edge_colors.append('red')
        else:
            g_i_edge_colors.append('black')


    nx.draw_networkx_nodes(g_i, pos=node_positions, node_size=6, alpha=0.6, node_color='lightgray',
                           with_labels=False, linewidths=0.1)
    nx.draw_networkx_edges(g_i, pos=node_positions, edge_color=g_i_edge_colors, alpha=0.8)

    plt.axis('off')
    ss= 'img{}.png'.format(i)
    plt.savefig(ss, dpi=120, bbox_inches='tight')
    plt.close()
    filenames.append(ss)


def make_video( movie, fps = 3):
    with imageio.get_writer( movie, mode='I', fps=fps ) as writer:
        for f in filenames:
            image = imageio.imread(f)
            writer.append_data(image)


make_video('png/movie.gif', fps=3)
