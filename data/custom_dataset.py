import pandas as pd
import torch
import os
from dgl.data import DGLDataset
from torch.utils.data.dataset import Dataset
import dgl
from collections import defaultdict
from dgl import save_graphs, load_graphs
import re
import concurrent.futures
from PIL import Image


def find_classes(_dir):
    classes = [d for d in os.listdir(_dir) if os.path.isdir(os.path.join(_dir, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx


def check_nan_inf(tensor):
    return torch.isnan(tensor).any() or torch.isinf(tensor).any()


class IfcSchemaGraphDataSet(DGLDataset):
    def __init__(self, name, edge_file_path, node_file_path, cache_path, in_feat, edge_direction, seed):
        self.node_types = None
        self.len = None
        self.g = None
        self.edge_file_path = edge_file_path
        self.node_file_path = node_file_path
        self._cache_path = cache_path
        self.in_feat = in_feat
        self.edge_direction = edge_direction
        self.seed = seed

        super(IfcSchemaGraphDataSet, self).__init__(name)

    def process(self):
        class_nodes = {}
        attribute_nodes = {}
        type_nodes = {}
        value_nodes = {}

        nodes = pd.read_csv(self.node_file_path)
        for _, row in nodes.iterrows():
            node_type, node_id, item = row
            if node_type == 'class_node':
                class_nodes[node_id] = item
            elif node_type == 'attribute_node':
                attribute_nodes[node_id] = item
            elif node_type == 'type_node':
                type_nodes[node_id] = item
            elif node_type == 'value_node':
                value_nodes[node_id] = item
        self.node_types = {'class_node': class_nodes, 'attribute_node': attribute_nodes, 'type_node': type_nodes, 'value_node': value_nodes}

        edges = pd.read_csv(self.edge_file_path)
        edge_dict = defaultdict(list)

        for _, row in edges.iterrows():
            src_type, src_id, dst_type, dst_id, edge_type = row
            edge_dict[(src_type, edge_type, dst_type)].append((src_id, dst_id))

        # if self.edge_direction == 'REVERSE':
        #     # 添加自指边
        #     value_node_ids = list(value_nodes.keys())
        #     if value_node_ids:
        #         edge_dict[('value_node', 'self_loop', 'value_node')] = [(node_id, node_id) for node_id in value_node_ids]

        graph = dgl.heterograph({etype: (torch.tensor(list(zip(*edata))[0]), torch.tensor(list(zip(*edata))[1])) for etype, edata in edge_dict.items()})
        self.len = len(edge_dict)

        torch.manual_seed(self.seed)
        for edge_type in graph.canonical_etypes:
            graph.edges[edge_type].data['feature'] = torch.randn(graph.number_of_edges(edge_type), self.in_feat)

        for ntype in graph.ntypes:
            graph.nodes[ntype].data['features'] = torch.randn(graph.num_nodes(ntype), self.in_feat)

        for ntype in graph.ntypes:
            if check_nan_inf(graph.nodes[ntype].data['features']):
                print("NaN or Inf found in node features of type {}".format(ntype))
        for edge_type in graph.canonical_etypes:
            if check_nan_inf(graph.edges[edge_type].data['feature']):
                print("NaN or Inf found in edge features of type {}".format(edge_type))

        if self.edge_direction == 'REVERSE':
            self.g = dgl.reverse(graph, copy_ndata=True, copy_edata=True)
        self.g = graph

    def has_cache(self):
        graph_path = os.path.join(self._cache_path, self.edge_direction + '_IfcSchemaGraph.bin')
        return os.path.exists(graph_path)

    def save(self):
        os.makedirs(self._cache_path, exist_ok=True)
        graph_path = os.path.join(self._cache_path, self.edge_direction + '_IfcSchemaGraph.bin')
        save_graphs(graph_path, self.g)

    def load(self):
        graph_path = os.path.join(self._cache_path, self.edge_direction + '_IfcSchemaGraph.bin')
        self.g = load_graphs(graph_path)[0][0]

    def __getitem__(self, index):
        assert index == 0, "这个数据集只有一个图"
        return self.g

    def __len__(self):
        return self.len


class IfcFileGraphsDataset(DGLDataset):
    def __init__(self, name, root, data_type, c_dir, isg, edge_direction):
        self.root = root
        self.classes, self.class_to_idx = find_classes(root)
        self.data_type = data_type
        self._cache_path = c_dir
        self.model_flag = []
        self.graphs = []
        self.labels = []
        self.isg = isg
        self.edge_direction = edge_direction
        self.metagraph = [('class_node', 'hasAttribute', 'attribute_node'),

                          ('attribute_node', 'hasValue', 'class_node'),
                          ('attribute_node', 'hasValue', 'type_node'),
                          ('attribute_node', 'hasValue', 'value_node'),

                          ('type_node', 'hasValue', 'class_node'),
                          ('type_node', 'hasValue', 'type_node'),
                          ('type_node', 'hasValue', 'value_node'),

                          ('class_node', 'selfLoop', 'class_node'),
                          ('attribute_node', 'selfLoop', 'attribute_node'),
                          ('type_node', 'selfLoop', 'type_node'),
                          ('value_node', 'selfLoop', 'value_node')]

        self.re_metagraph = [('attribute_node', 'hasAttribute', 'class_node'),

                             ('class_node', 'hasValue', 'attribute_node'),
                             ('type_node', 'hasValue', 'attribute_node'),
                             ('value_node', 'hasValue', 'attribute_node'),

                             ('class_node', 'hasValue', 'type_node'),
                             ('type_node', 'hasValue', 'type_node'),
                             ('value_node', 'hasValue', 'type_node'),

                             ('class_node', 'selfLoop', 'class_node'),
                             ('attribute_node', 'selfLoop', 'attribute_node'),
                             ('type_node', 'selfLoop', 'type_node'),
                             ('value_node', 'selfLoop', 'value_node')]

        super(IfcFileGraphsDataset, self).__init__(name)

    @staticmethod
    def _is_number(obj: str) -> bool:
        pattern = r'^[-+]?[0-9]*\.[0-9]+([eE][-+]?[0-9]+)?$'
        return bool(re.match(pattern, obj))

    @staticmethod
    def _is_integer(obj: str) -> bool:
        # 匹配整数的正则表达式
        pattern = r'^[-+]?[0-9]+$'
        return bool(re.match(pattern, obj))

    @staticmethod
    def _is_bool(obj: str) -> bool:
        return obj in {'True', 'False'}

    @staticmethod
    def _sigmoid(x: float) -> float:
        return 1 / (1 + torch.exp(-torch.tensor(x)))

    def process(self):
        results = {}
        # <root>/<label>/<type>/<item>/<graph>/<data>.csv
        for label in os.listdir(self.root):
            label_path_list = []
            type_path = os.path.join(self.root, label, self.data_type)
            for item in os.listdir(str(type_path)):
                graph_path = os.path.normpath(os.path.join(str(type_path), str(item), "graph"))
                label_path_list.append((label, graph_path))
            num_processes = 4
            while True:
                results_temp = {}  # 存储临时结果
                failed_files = []  # 记录失败的文件
                with concurrent.futures.ProcessPoolExecutor(max_workers=num_processes) as executor:
                    future_to_file = {executor.submit(self._process_label_path, file): file for file in label_path_list}
                    for future in concurrent.futures.as_completed(future_to_file):
                        try:
                            result = future.result()
                            results_temp[label] = results_temp.get(label, []) + [result]
                        except Exception as exc:
                            file = future_to_file[future]
                            print('{},{}'.format(file, exc))
                            failed_files.append(file)
                # 更新最终结果
                if label in results and results[label]:  # 检查是否存在且有值
                    results[label] += results_temp.get(label, [])  # 拼接结果
                else:
                    results[label] = results_temp.get(label, [])  # 赋值
                # 如果没有失败的文件，退出循环
                if not failed_files:
                    print("{}更新完了{}".format(len(results.get(label, [])), len(label_path_list)))
                    break
                label_path_list = [(label, file) for _, file in failed_files]

        # with open('_results.pkl', 'wb') as pkl_file:
        #      pickle.dump(results, pkl_file)

        for _, result in results.items():
            for item in result:
                model_flag, g, label = item[0]
                self.graphs.append(g)
                self.labels.append(label)
                self.model_flag.append(model_flag)

    def _process_label_path(self, label_path):
        label, _path = label_path
        graphs = []
        a = ''
        for root, dirs, files in os.walk(_path):
            node_file_path = ''
            edge_file_path = ''
            bin_file_path = ''
            for file in files:
                if file.endswith('node.csv'):
                    node_file_path = os.path.join(root, file)
                elif file.endswith('edge.csv'):
                    edge_file_path = os.path.join(root, file)
                elif file.endswith(self.isg):
                    bin_file_path = os.path.join(root, file)

            if node_file_path != '' and edge_file_path != '':
                parts = root.split('\\')
                _model_flag = parts[2]+parts[4]
                _shared_embedding_information = torch.load(str(bin_file_path), weights_only=False)
                graph = self._load_hetero_graph_from_csv(edge_file_path, node_file_path, label, _shared_embedding_information, _model_flag)
                graphs.append(graph)
                a = node_file_path
        print(' {}已完成'.format(a))
        return graphs

    def _load_hetero_graph_from_csv(self, edge_file_path, node_file_path, label, _shared_embedding_information, model_flag):
        print('{}开始'.format(node_file_path))
        # 定义所有可能的边类型
        all_edge_types = self.metagraph

        # 初始化包含所有可能边类型的字典，所有边类型对应的值初始为空列表
        edge_dict = {etype: ([], []) for etype in all_edge_types}
        # edge_dict = {etype: ([], []) for etype in all_edge_types if etype[1] != 'selfLoop'}  # 去掉自环

        edges = pd.read_csv(edge_file_path, header=None)
        # 根据CSV中的数据填充边字典
        for _, row in edges.iterrows():
            src_type, src_id, dst_type, dst_id, edge_type = row

            # # 去掉自环
            # if edge_type == 'selfLoop':
            #     continue
            # else:
            #     edge_key = (src_type, edge_type, dst_type)
            #     edge_dict[edge_key][0].append(src_id)
            #     edge_dict[edge_key][1].append(dst_id)

            # 带有自环
            edge_key = (src_type, edge_type, dst_type)
            edge_dict[edge_key][0].append(src_id)
            edge_dict[edge_key][1].append(dst_id)

        # 创建异构图
        g = dgl.heterograph(edge_dict)

        # 按照IFCSchema嵌入表示结果初始化特征
        nodes_infor = pd.read_csv(node_file_path, header=None)
        node_feat_dict = defaultdict(dict)
        for seed_flag, row in nodes_infor.iterrows():
            node_type, node_id, item = row

            # # w/o ifcSchem
            # torch.manual_seed(int(seed_flag))
            # features = torch.rand_like(_shared_embedding_information['type_node']['INTEGER'])

            # with ifcSchema
            if node_type == 'class_node':
                features = _shared_embedding_information['class_node'][item]
            elif node_type == 'attribute_node':
                features = _shared_embedding_information['attribute_node'][item]
            elif node_type == 'type_node':
                if item == 'TUPLE':
                    features = _shared_embedding_information['type_node']['LIST']
                elif item == 'TUPLETUPLE':
                    features = _shared_embedding_information['type_node']['LISTLIST']
                else:
                    features = _shared_embedding_information['type_node'][item]
            elif node_type == 'value_node':
                if item.upper() in _shared_embedding_information['value_node'].keys():
                    features = _shared_embedding_information['value_node'][item.upper()]
                elif self._is_integer(str(item)):
                    scaled_value = self._sigmoid(float(item))
                    features = scaled_value * _shared_embedding_information['type_node']['INTEGER']
                elif self._is_number(str(item)):
                    scaled_value = self._sigmoid(float(item))
                    features = scaled_value * _shared_embedding_information['type_node']['REAL']
                elif self._is_bool(str(item)):
                    scaled_value = 1.0 if item == 'True' else 0.0
                    features = scaled_value * _shared_embedding_information['type_node']['BOOLEAN']
                else:
                    features = _shared_embedding_information['type_node']['STRING']
            else:
                print(node_type, node_id, item)
                features = _shared_embedding_information[node_type][item]

            node_feat_dict[node_type][node_id] = features
        for ntype in node_feat_dict:
            node_ids = list(node_feat_dict[ntype].keys())
            features_list = [node_feat_dict[ntype][nid] for nid in node_ids]
            feats = torch.stack(features_list)
            if g.num_nodes(ntype) != len(feats):
                print("不相等：Node type: {}, Number of nodes: {}, Number of features: {}".format(ntype, g.num_nodes(ntype), len(feats)))
            g.nodes[ntype].data['features'] = feats

        label = self.class_to_idx[label]
        gra = (model_flag, g, label)
        return gra

    def save(self):
        os.makedirs(self._cache_path, exist_ok=True)
        graph_path = os.path.join(self._cache_path, self.data_type, 'graphs.bin')
        label_path = os.path.join(self._cache_path, self.data_type, 'labels.pt')
        flag_path = os.path.join(self._cache_path, self.data_type, 'flags.pt')
        sorted_data = sorted(zip(self.model_flag, self.graphs, self.labels), key=lambda x: x[0])

        self.model_flag, self.graphs, self.labels = map(list, zip(*sorted_data))

        save_graphs(str(graph_path), self.graphs)
        torch.save(self.labels, str(label_path))
        torch.save(self.model_flag, str(flag_path))

    def load(self):
        graph_path = os.path.join(self._cache_path, self.data_type, 'graphs.bin')
        label_path = os.path.join(self._cache_path, self.data_type, 'labels.pt')
        flag_path = os.path.join(self._cache_path, self.data_type, 'flags.pt')
        self.graphs, _ = load_graphs(str(graph_path))
        self.labels = torch.load(str(label_path), weights_only=True)
        self.model_flag = torch.load(str(flag_path), weights_only=True)

    def has_cache(self):
        graph_path = os.path.join(self._cache_path, self.data_type, 'graphs.bin')
        label_path = os.path.join(self._cache_path, self.data_type, 'labels.pt')
        flag_path = os.path.join(self._cache_path, self.data_type, 'flags.pt')
        return os.path.exists(graph_path) and os.path.exists(label_path) and os.path.exists(flag_path)

    def get_model_flag(self, index):
        model_flag = self.model_flag[index]
        return model_flag

    def __getitem__(self, index):
        g = self.graphs[index]
        if self.edge_direction == 'REVERSE':
            g = dgl.reverse(g, copy_ndata=True, copy_edata=True)

        return g, self.labels[index]

    def __len__(self):
        return len(self.graphs)


class MultiViewDataSet(Dataset):
    def __init__(self, root, data_type, view_name, data_transform=None, target_transform=None):
        self.x = []
        self.y = []
        self.root = root
        self.classes, self.class_to_idx = find_classes(root)
        self.data_transform = data_transform
        self.target_transform = target_transform

        # <root>/<label>/<type>/<item>/<view>/<data>.png
        for label in os.listdir(root):
            item_path = os.path.join(root, label, data_type)
            for item in os.listdir(str(item_path)):
                views = []
                view_path = os.path.join(str(item_path), str(item), view_name)
                for view in os.listdir(str(view_path)):
                    views.append(os.path.join(view_path, view))
                self.x.append(views)
                self.y.append(self.class_to_idx[label])

    def __getitem__(self, index):
        org_views = self.x[index]
        views = []
        for view in org_views:
            im = Image.open(str(view))
            im = im.convert('RGB')
            if self.data_transform is not None:
                im = self.data_transform(im)
            views.append(im)
        return views, self.y[index]

    def __len__(self):
        return len(self.x)


