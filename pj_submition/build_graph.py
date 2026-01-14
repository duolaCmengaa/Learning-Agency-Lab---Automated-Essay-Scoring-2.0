import pandas as pd
import numpy as np
import networkx as nx
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import scipy.sparse as sp
import argparse
import os

def build_graph(data_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 读取数据集
    df = pd.read_csv(data_path)
    documents = df['full_text'].values
    labels = df['score'].values
    ids = df['essay_id'].values
    
    # 计算TF-IDF
    vectorizer = TfidfVectorizer(max_features=5000)
    tfidf_matrix = vectorizer.fit_transform(documents)
    vocab = vectorizer.get_feature_names_out()
    print('TF-IDF calculated')
    
    # 计算文档相似度
    cosine_sim_doc = cosine_similarity(tfidf_matrix)
    print('Doc cosine similarity calculated')
    
    # 计算词汇相似度
    cosine_sim_vocab = cosine_similarity(tfidf_matrix.T)
    print('Word cosine similarity calculated')
    
    # 构建图
    graph = nx.Graph()
    
    # 添加文档节点
    for i, doc in enumerate(documents):
        graph.add_node(i, id=ids[i], label=labels[i], type='doc')



    # 添加词节点
    word_id_map = {word: idx for idx, word in enumerate(vocab)}
    for word, idx in word_id_map.items():
        graph.add_node(len(documents) + idx, word=word, type='word')

    print('Nodes created')
    
    # 添加文档-词边
    # for i, doc in enumerate(documents):
    #     words = doc.split()
    #     word_set = set(words)
    #     for word in word_set:
    #         if word in word_id_map:
    #             word_id = word_id_map[word]
    #             graph.add_edge(i, len(documents) + word_id, weight=1.0)
    
    
    # 添加文档-词边，权重为TF-IDF值
    num_docs = len(documents)
    for doc_id in range(num_docs):
        doc_vector = tfidf_matrix[doc_id]
        for word_id, tfidf_value in zip(doc_vector.indices, doc_vector.data):
            graph.add_edge(doc_id + 1, num_docs + word_id + 1, weight=tfidf_value)
    print('Doc-word edge added')


    # 添加词-词边
    for i in range(len(vocab)):
        for j in range(i + 1, len(vocab)):
            sim = cosine_sim_vocab[i, j]
            if sim > 0.1:  # 设置一个相似度阈值
                graph.add_edge(len(documents) + i, len(documents) + j, weight=sim)
    print('Word-word edge added')
    

    # 将图转换为稀疏矩阵
    adj = nx.adjacency_matrix(graph)
    features = tfidf_matrix
    labels = torch.tensor(labels)
    print('Finished!')
    
    return adj, features, labels

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True, help='Path to the dataset CSV file.')
    args = parser.parse_args()
    
    adj, features, labels = build_graph(args.data_path)
    
    # 保存图和特征矩阵用于后续使用
    if not os.path.exists("./graph"):
        os.makedirs("./graph", exist_ok=True)
    sp.save_npz('./graph/adj.npz', adj)  # 稀疏的邻接矩阵，表示图结构
    sp.save_npz('./graph/features.npz', features)  # 特征矩阵，表示每个文档的TF-IDF特征
    torch.save(labels, './graph/labels.pt')  # 标签，表示每个文档的类别标签
