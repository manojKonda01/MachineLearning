#!/usr/bin/env python
# coding: utf-8

# In[1]:


import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
from operator import itemgetter
import seaborn as sns
import numpy as np
from networkx.algorithms import node_classification
from networkx.algorithms import community


# In[2]:


import warnings 
warnings.filterwarnings('ignore')


# In[3]:


# Reading datasets and importing data
G = nx.Graph(day="Stackoverflow")
df_nodes = pd.read_csv("dataset/stack_network_nodes.csv")
df_edges = pd.read_csv("dataset/stack_network_links.csv")
for index, row in df_nodes.iterrows():
    G.add_node(row['name'], group=row['group'], nodesize=row['nodesize'])

for index, row in df_edges.iterrows():
    G.add_weighted_edges_from([(row['source'], row['target'], row['value'])])


# In[4]:


#Nodes dataframe
df_nodes.head(10)


# In[5]:


# edges dataframe
df_edges.head(10)


# In[6]:


#general info of network
print(nx.info(G))


# In[7]:


print("\nList of all {} nodes present\n{}".format(len(G.nodes()), G.nodes()))


# In[8]:


# check whether our network is connected or not connected
if nx.is_connected(G):
    print('Connected Graph')
else:
    print("Not connected")


# In[9]:


# Network Density
print("\nNetwork density:", nx.density(G))
# network density of 0.037376 says that it is not completely connected.
# May not all the nodes are connected


# In[10]:


# diameter of largest component
components = nx.connected_components(G)
largest_component = max(components, key=len)
print(largest_component)
subgraph = G.subgraph(largest_component)
diameter = nx.diameter(subgraph)
print("\nNetwork diameter of largest component:", diameter)


# In[11]:


# Triadic closure(Local Clustering coefficient). This is one of the link prediction methods
triadic_closure = nx.transitivity(G)
print("\nTriadic closure:", triadic_closure)


# # Centrality

# In[12]:


# Degree Centrality
# Top three nodes having the highest Degree Centrality
def find_nodes_with_highest_deg_cent(G):
    # Compute the degree centrality of G: deg_cent
    deg_cent = nx.degree_centrality(G)
    # Compute the maximum degree centrality: max_dc
    max_1_dc = max(list(deg_cent.values()))
    max_2_dc = list(sorted(deg_cent.values()))[-2]
    max_3_dc = list(sorted(deg_cent.values()))[-3]

    maxnode1 = set()
    maxnode2 = set()
    maxnode3 = set()

    # Iterate over the degree centrality dictionary
    for k, v in deg_cent.items():

        # Check if the current value has the maximum degree centrality
        if v == max_1_dc:
            # Add the current node to the set of nodes
            maxnode1.add(k)
        if v == max_2_dc:
            # Add the current node to the set of nodes
            maxnode2.add(k)
        if v == max_3_dc:
            # Add the current node to the set of nodes
            maxnode3.add(k)

    return maxnode1, maxnode2, maxnode3
top_deg_dc, top2_deg_dc, top3_deg_dc = find_nodes_with_highest_deg_cent(G)
print("\nTop three nodes having the highest degree centrality :", top_deg_dc, top2_deg_dc,
      top3_deg_dc)


# # Degree Centrality

# In[13]:


from collections import Counter
degree_dic = dict(G.degree(G.nodes()))
nx.set_node_attributes(G, degree_dic, 'degree')

degree_hist = pd.DataFrame({"degree": list(degree_dic.values()),
                            "Nodes": list(degree_dic.keys())})
plt.figure(figsize=(10,10))
clrs = ['darkblue' if (x < max(degree_dic.values())) else 'red' for x in degree_dic.values() ]
sns.barplot(x = 'degree', y = 'Nodes', 
              data = degree_hist, 
              palette=clrs)
plt.ylabel('Node', fontsize=30)
plt.xlabel('Degree', fontsize=30)
plt.tick_params(axis='both', which='major',labelsize=10)

plt.show()
plt.savefig('nodedegree.png')
# clearly 'jquery' has highest degree


# In[14]:


# degree of each node
degree=G.degree(G.nodes())
for i in degree:
    print(i,end=" ")


# In[15]:


# Degree centrality
dc_dict = nx.degree_centrality(G)
nx.set_node_attributes(G, dc_dict, 'degree')
sorted_dc = sorted(dc_dict.items(), key=itemgetter(1), reverse=True)
print("Order of nodes According to their importance by using Degree centrality")
for i in sorted_dc:
    print(i)


# In[16]:


# TOp 3 nodes with highest Degree centrality
print("\nTop three nodes having highest Degree centrality")
for i in sorted_dc[:3]:
    print(i)


# # Eigenvector Centrality

# In[17]:


# Eigenvector Centrality
eigenvector_dict = nx.eigenvector_centrality(G)
nx.set_node_attributes(G, eigenvector_dict, 'eigenvector')
sorted_eigenvector = sorted(eigenvector_dict.items(), key=itemgetter(1), reverse=True)
print("Order of nodes According to their importance by using EigenVector Centrality")
for i in sorted_eigenvector:
    print(i)


# In[18]:


# Top 3 nodes with highest Eigenvector centrality
print("\nTop three nodes having highest Eigenvector centrality")
for i in sorted_eigenvector[:3]:
    print(i)


# # Betweenness Centrality

# In[19]:


# Betweenness Centrality
betweenness_dict = nx.betweenness_centrality(G)
nx.set_node_attributes(G, betweenness_dict, 'betweenness')
sorted_betweenness = sorted(betweenness_dict.items(), key=itemgetter(1), reverse=True)
print("Order of nodes According to their importance by using Betweenness Centrality")
for i in sorted_betweenness:
    print(i)


# In[20]:


# Top 3 nodes with highest Betweenness centrality
print("\nTop three nodes having highest Betweenness centrality")
for b in sorted_betweenness[:3]:
    print(b)


# In[21]:


# shortest path between nodes
nx.shortest_path(G,'jquery','c#')


# In[22]:


nx.shortest_path(G,'jquery','redux')


# In[23]:


nx.shortest_path(G,'linq','xml')


# In[24]:


# from all the 3 centralities common node is 'jquery'
# since it is not connected
# shortest path lengths from highest centrality node 'jquery' are
shortlength_jquery=nx.shortest_path_length(G,'jquery')
shortlength_jquery


# # Building a subgroup
# - We can find the distance of a node from every other node in the network using breadth-first search algorithm, starting from that node. networkX provides the function bfs_tree to do it.

# In[25]:


sub1 = nx.bfs_tree(G,'jquery')


# In[26]:


sub2 = nx.bfs_tree(G,'css')


# In[27]:


# Subgroup (an oriented tree constructed from of a breadth-first-search starting at "jquery")
plt.figure(figsize=(25, 25))
options = {
    'edge_color': '#BAB0AD',
    'width': 1,
    'with_labels': True,
    'font_weight': 'normal',
    'font_size': 15,
    'style': 'dashed'
}
sizes = [G.nodes[node]['nodesize'] * 10 for node in G]
nx.draw_networkx(sub1, pos=nx.spring_layout(G, k=0.25, iterations=50), **options)
nx.draw_networkx(sub1.subgraph('jquery'), pos=nx.spring_layout(G, k=0.25, iterations=50),node_color='red', **options)
ax = plt.gca()
ax.collections[0].set_edgecolor("#555555")
plt.show()


# In[28]:


# Subgroup (an oriented tree constructed from of a breadth-first-search starting at "css")
plt.figure(figsize=(25, 25))
options = {
    'edge_color': '#BAB0AD',
    'width': 1,
    'with_labels': True,
    'font_weight': 'normal',
    'font_size': 15,
    'style': 'dashed'
}
sizes = [G.nodes[node]['nodesize'] * 10 for node in G]
nx.draw_networkx(sub2, pos=nx.spring_layout(G, k=0.25, iterations=50), **options)
nx.draw_networkx(sub2.subgraph('css'), pos=nx.spring_layout(G, k=0.25, iterations=50),node_color='red', **options)
ax = plt.gca()
ax.collections[0].set_edgecolor("#555555")
plt.show()


# In[29]:


# Nodes which are connected to important node
print(nx.node_connected_component(G,'jquery'))
print("\nTotal {} nodes are connected with main important 'jquery' node".format(len(nx.node_connected_component(G,'jquery'))))


# In[30]:


# Nodes which are not connected to important node
print(G.nodes()-list(nx.node_connected_component(G,'jquery')))


# # Node Classification

# In[31]:


# harmonic_function
G.nodes['angular']['label']='web API framework'
G.nodes['css']['label']='web design '
G.nodes['c++']['label']='programming language'
G.nodes['git']['label']='command shell'
G.nodes['linux']['label']='OS'
G.nodes['qt']['label']='GUI'
G.nodes['hibernate']['label']='database'
classs = node_classification.harmonic_function(G)
nodes=list(G.nodes())
node_class={nodes[i]:classs[i] for i in range(len(nodes))}
for i in node_class:
    print(i,"-------->",node_class.get(i))


# # Link prediction
# - Jaccard Coefficient

# In[32]:


# jaccard coefficient
threshold_j=0.45
jaccard=list(nx.jaccard_coefficient(G))
for i in jaccard:
    if i[2]>threshold_j:
        print(i)


# - Resource Allocation Index
# (predict missing links, similarity between two nodes)

# In[33]:


# Resource Allocation Index
threshold_RAI=0.45
RAI=list(nx.resource_allocation_index(G))
for i in RAI:
    if i[2]>threshold_RAI:
        print(i)


# - Adamic Adar Index  
# 1. predict missing links in a Network, according to the amount of shared links between two nodes

# In[34]:


# Adamic adar Index
threshold_AAI=1.45
AAI=list(nx.adamic_adar_index(G))
for i in AAI:
    if i[2]>threshold_AAI:
        print(i)


# - Preferential Attachment 
# 1. The probability of generating a new link of node u is directly proportional to the degree of the node

# In[35]:


#preferential Attachment
degree_dic = dict(G.degree(G.nodes()))
minn=min(degree_dic.values())
maxx=max(degree_dic.values())
avg=sum(degree_dic.values())/len(degree_dic)
print("Minimum degree = {}\nMaximum degree={}\nAverage Degree={}".format(minn,maxx,avg))


# In[36]:


thres_d=maxx*avg
PA=list(nx.preferential_attachment(G))
for i in PA:
    if i[2]>thres_d:
        print(i)


# # Community Detection

# - Greedy Modularity Community

# In[37]:


# Greedy Modularity
communities = community.greedy_modularity_communities(G)
for i in communities:
    print(i,"\n")


# In[38]:


modularity_dict = {} # Create a blank dictionary
for i,c in enumerate(communities): # Loop through the list of communities, keeping track of the number for the community
    for name in c: # Loop through each person in a community
        modularity_dict[name] = i # Create an entry in the dictionary for the person, where the value is which group they belong to.

# Now you can add modularity information like we did the other metrics
nx.set_node_attributes(G, modularity_dict, 'modularity')


# In[39]:


# First get a list of just the nodes in that class
class0 = [n for n in G.nodes() if G.nodes[n]['modularity'] == 0]

# Then create a dictionary of the eigenvector centralities of those nodes
class0_eigenvector = {n:G.nodes[n]['eigenvector'] for n in class0}

# Then sort that dictionary and print the first 5 results
class0_sorted_by_eigenvector = sorted(class0_eigenvector.items(), key=itemgetter(1), reverse=True)

print("Modularity Class 0 Sorted by Eigenvector Centrality:")
for node in class0_sorted_by_eigenvector[:5]:
    print("Name:", node[0], "| Eigenvector Centrality:", node[1])


# In[40]:


for i,c in enumerate(communities): # Loop through the list of communities
    if len(c) > 2: # Filter out modularity classes with 2 or fewer nodes
        print('Class '+str(i)+':', list(c),"\n") # Print out the classes and their members


# - label_propagation_communities

# In[41]:


# Label Propagation communities
label_prop_comm=community.label_propagation_communities(G)
j=0
for i in label_prop_comm:
    print("Class [{}] ----> {}".format(j,i))
    j+=1


# # Drawing Networks

# In[42]:


color_map = {1: '#f09494', 2: '#eebcbc', 3: '#72bbd0', 4: '#91f0a1', 5: '#629fff', 6: '#bcc2f2',
             7: '#eebcbc', 8: '#f1f0c0', 9: '#d2ffe7', 10: '#caf3a6', 11: '#ffdf55', 12: '#ef77aa',
             13: '#d6dcff', 14: '#d2f5f0', 15: '#2B2B40', 16: '#e6bbaa', 17: '#c158fd'}

plt.figure(figsize=(25, 25))
options = {
    'edge_color': '#BAB0AD',
    'width': 1,
    'with_labels': True,
    'font_weight': 'normal',
    'font_size': 15,
    'style': 'dashed'
}
colors = [color_map[G.nodes[node]['group']] for node in G]
sizes = [G.nodes[node]['nodesize'] * 20 for node in G]

"""
Using the spring layout : 
- k controls the distance between the nodes and varies between 0 and 1
- iterations is the number of times simulated annealing is run
default k=0.1 and iterations=50
"""

# nx.spring_layout(G, k=0.25, iterations=50)
nx.draw(G, node_color=colors, node_size=sizes, pos=nx.spring_layout(G, k=1, iterations=50), **options)
ax = plt.gca()
ax.collections[0].set_edgecolor("#555555")
plt.show()
plt.savefig("Network.png")

