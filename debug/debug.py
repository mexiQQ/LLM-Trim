# import networkx as nx
# import matplotlib.pyplot as plt

# # 创建一个空图
# G = nx.Graph()

# # 添加边，每个边由一对顶点定义
# edges = [(1, 2), (1, 3), (2, 4), (3, 4)]  # 例如
# G.add_edges_from(edges)

# # 绘制图
# nx.draw(G, with_labels=True, font_weight='bold')

# # 显示图
# plt.savefig("test.png")

import networkx as nx
import matplotlib.pyplot as plt

# 创建图
G = nx.Graph()

# 添加带有权重的边
G.add_edge('A', 'B', weight=2)
G.add_edge('B', 'C', weight=4)
G.add_edge('A', 'C', weight=1)

# 用Spring布局来反映边的权重，这将尝试使边的长度与权重成反比
pos = nx.spring_layout(G, weight='weight')

# 绘制节点
nx.draw_networkx_nodes(G, pos)

# 绘制边
nx.draw_networkx_edges(G, pos)

# 标签
nx.draw_networkx_labels(G, pos)

# 显示图形
plt.savefig("figures/test.png")