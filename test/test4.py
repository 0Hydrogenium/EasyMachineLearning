import networkx as nx
import matplotlib.pyplot as plt

def hmm_visual(a, b, c):
    # 以下是一个简单的例子，你需要替换为你的实际概率矩阵数据
    transition_matrix = {
        'A': {'A': 0.8, 'B': 0.2},
        'B': {'A': 0.3, 'B': 0.7}
    }

    # 创建有向图
    G = nx.DiGraph()

    # 添加节点
    for state in transition_matrix:
        G.add_node(state)

    # 添加边和权重
    for from_state, to_states in transition_matrix.items():
        for to_state, probability in to_states.items():
            G.add_edge(from_state, to_state, weight=probability)

    # 绘制网络图
    pos = nx.spring_layout(G)  # 使用 spring_layout 布局算法
    labels = nx.get_edge_attributes(G, 'weight')

    # 使用下面的两行代码绘制图形
    fig, ax = plt.subplots()
    nx.draw(G, pos, with_labels=True, node_size=700, node_color="skyblue", font_size=10, font_color="black", font_weight="bold", edge_color="gray", ax=ax)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels, font_color='red', ax=ax)

    plt.show()
