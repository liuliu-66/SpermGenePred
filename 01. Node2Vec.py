import pandas as pd
import networkx as nx
from node2vec import Node2Vec
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

ppi_data = pd.read_csv('PPI.txt', sep='\t') 

G = nx.Graph()
for _, row in ppi_data.iterrows():
    gene1 = row['preferred_name1_converted']
    gene2 = row['preferred_name2_converted']
    G.add_edge(gene1, gene2)

node2vec = Node2Vec(G, dimensions=64, walk_length=30, num_walks=200, workers=4)
model = node2vec.fit()

gene_embeddings = {}
for gene in G.nodes():
    gene_embeddings[gene] = model.wv[gene]

embedding_df = pd.DataFrame(gene_embeddings).T
embedding_df['genes'] = embedding_df.index
