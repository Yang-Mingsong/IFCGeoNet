import os
import dgl
import numpy as np
import torch
import wandb
import torch.nn.functional as F
import sklearn.metrics as metrics
from tqdm import tqdm


def info_nce_loss(anchor, positive, negative, temperature):
    # 对锚点、正样本、负样本进行 L2 归一化
    anchor = F.normalize(anchor, dim=1)
    positive = F.normalize(positive, dim=1)
    negative = F.normalize(negative, dim=1)

    # 计算正样本对和负样本对的相似性（余弦相似度）
    pos_similarity = torch.sum(anchor * positive, dim=-1) / temperature  # 正样本对
    neg_similarity = torch.sum(anchor * negative, dim=-1) / temperature  # 负样本对

    # 创建相似度矩阵：将正样本和负样本的相似性拼接
    logits = torch.cat([pos_similarity.unsqueeze(1), neg_similarity.unsqueeze(1)], dim=1)  # [batch_size, 2]

    # 计算 InfoNCE 损失（交叉熵）
    labels = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)
    loss = F.cross_entropy(logits, labels)
    return loss


def compute_softmax_outputs(mvcnn, rgcn, device, mvcnn_test_dataloader, rgcn_test_dataloader):
    softmax_mvcnn, softmax_rgcn, labels = [], [], []
    tqdm_batch = tqdm(zip(mvcnn_test_dataloader, rgcn_test_dataloader))
    for (image_batches, img_label_batches), (graph_batches, gra_label_batches) in tqdm_batch:
        with torch.no_grad():
            # 数据准备
            img_inputs = np.stack(image_batches, axis=1)  # 12,12,3,224,224
            img_inputs = torch.from_numpy(img_inputs)
            N, V, C, H, W = img_inputs.size()
            img_inputs = img_inputs.view(-1, C, H, W)
            img_inputs = img_inputs.to(device)

            batched_graph = graph_batches.to(device)
            features = {}
            for ntype in batched_graph.ntypes:
                features[ntype] = batched_graph.nodes[ntype].data['features']
            if torch.equal(img_label_batches, gra_label_batches):
                label_batches = img_label_batches.to(device)
            else:
                print("大错特错")
                print("img_label_batches{}".format(img_label_batches))
                print("gra_label_batches{}".format(gra_label_batches))
                label_batches = gra_label_batches.to(device)

            targets = label_batches.to(device)

            # compute output
            image_features = mvcnn(img_inputs)
            graph_features = rgcn(batched_graph)

            softmax_img = torch.softmax(image_features, dim=1)
            softmax_graph = torch.softmax(graph_features, dim=1)
            softmax_mvcnn.append(softmax_img.cpu().numpy())
            softmax_rgcn.append(softmax_graph.cpu().numpy())
            labels.append(targets.cpu().numpy())

    return np.concatenate(softmax_mvcnn, axis=0), np.concatenate(softmax_rgcn, axis=0), np.concatenate(labels, axis=0)


class HGATTrainer(object):
    def __init__(self, model, graph, optimizer, device, edge_direction, _seed):
        self.node_features = None
        self.model = model
        self.graph = graph
        self.device = device
        self.optimizer = optimizer
        self.edge_direction = edge_direction
        self.positive_pairs = []
        self.negative_pairs = []
        if edge_direction == "REVERSE":
            self.re_isg_positive_negative_sample(_seed)
        else:
            self.isg_positive_negative_sample(_seed)
        self.best_loss = 0.0

    def has_edge_between_nodes(self, src_node, dst_node):
        for edge_type in self.graph.canonical_etypes:
            try:
                is_connected = self.graph.has_edges_between(src_node, dst_node, edge_type)
                if is_connected is True:
                    return True
            except dgl.DGLError:
                return False
        return False

    def isg_positive_negative_sample(self, seed):
        for etype in self.graph.canonical_etypes:  # 遍历异构图的边类型
            src, dst = self.graph.edges(etype=etype)  # 以边为索引，找到所有起始节点集合src，终止节点集合dst
            pos_pars = list(zip(src.tolist(), dst.tolist(), np.repeat(etype[0], repeats=len(src)), np.repeat(etype[-1], repeats=len(src))))  # 将起始节点ID和终点ID匹配
            self.positive_pairs.extend(pos_pars)  # self.positive_pairs:确定具有连接的节点对

        def generate_class_class_pairs(src, dst, dst_type, positive_pairs):
            for src1, dst1, src_type1, dst_type1 in positive_pairs:
                if src1 == src and src_type1 == 'class_node' and dst_type1 == 'class_node' and dst1 != dst:
                    t = (dst, dst1, dst_type, dst_type1)
                    return t

        def generate_type_pairs(src, dst, dst_type, positive_pairs):
            for src1, dst1, src_type1, dst_type1 in positive_pairs:
                if src1 == src and src_type1 == 'type_node' and dst1 != dst:
                    t = (dst, dst1, dst_type, dst_type1)
                    return t

        def generate_attribute_value_pairs(src, dst, dst_type, positive_pairs):
            for src1, dst1, src_type1, dst_type1 in positive_pairs:
                if src1 == src and src_type1 == 'attribute_node' and dst_type1 == 'value_node' and dst1 != dst:
                    t = (dst, dst1, dst_type, dst_type1)
                    return t

        # extended_pairs = set()
        # for src, dst, src_type, dst_type in self.positive_pairs:
        #     generated = None
        #     if (src_type, dst_type) == ('class_node', 'class_node'):
        #         generated = generate_class_class_pairs(src, dst, dst_type, self.positive_pairs)
        #     elif src_type == 'type_node':
        #         generated = generate_type_pairs(src, dst, dst_type, self.positive_pairs)
        #     elif (src_type, dst_type) == ('attribute_node', 'value_node'):
        #         generated = generate_attribute_value_pairs(src, dst, dst_type, self.positive_pairs)
        #
        #     # 直接添加有效元素
        #     if generated and isinstance(generated, tuple) and len(generated) == 4:
        #         # 标准化排序并保留类型
        #         a, b = sorted(generated[:2])
        #         sorted_pair = (a, b) + generated[2:]
        #         extended_pairs.add(sorted_pair)
        #
        # self.positive_pairs.extend(extended_pairs)

        extended_pairs = set()
        for src, dst, src_type, dst_type in self.positive_pairs:
            if (src_type, dst_type) == ('class_node', 'class_node'):
                extended_pairs.add(generate_class_class_pairs(src, dst, dst_type, self.positive_pairs))
            elif src_type == 'type_node':
                extended_pairs.add(generate_type_pairs(src, dst, dst_type, self.positive_pairs))
            elif (src_type, dst_type) == ('attribute_node', 'value_node'):
                extended_pairs.add(generate_attribute_value_pairs(src, dst, dst_type, self.positive_pairs))
        filtered_set = set()
        for pair in extended_pairs:
            if pair is not None and isinstance(pair, tuple) and len(pair) == 4:
                filtered_set.add(pair)
        extended_pairs = filtered_set

        unique_pairs = set()
        # 用于存储要删除的元素
        to_remove = set()
        for pair in extended_pairs:
            # 将对进行排序
            sorted_pair = tuple(sorted(pair[:2])) + pair[2:]
            # 如果排序后的对已经在新集合中，则标记为要删除
            if sorted_pair in unique_pairs:
                to_remove.add(pair)
            else:
                # 否则，添加到新集合中
                unique_pairs.add(sorted_pair)
        # 从原始集合中删除重复元素
        extended_pairs -= to_remove
        self.positive_pairs.extend(extended_pairs)

        if seed is not None:
            np.random.seed(seed)
        for src, dst, src_type, dst_type in self.positive_pairs:
            neg_item = self.positive_pairs[np.random.choice(range(len(self.positive_pairs)))]
            # 随机挑选与当前节点类型不同，且不相连，且在negative中不重复的样本对，作为负样本
            while (neg_item[1] == dst or src_type == neg_item[-1] or (src, neg_item[1], src_type, neg_item[-1]) in self.negative_pairs or
                   self.has_edge_between_nodes(src, neg_item[1])):
                neg_item = self.positive_pairs[np.random.choice(range(len(self.positive_pairs)))]
            self.negative_pairs.append((src, neg_item[1], src_type, neg_item[-1]))

    def re_isg_positive_negative_sample(self, seed):

        for etype in self.graph.canonical_etypes:  # 遍历异构图的边类型
            src, dst = self.graph.edges(etype=etype)  # 以边为索引，找到所有起始节点集合src，终止节点集合dst
            pos_pars = list(zip(src.tolist(), dst.tolist(), np.repeat(etype[0], repeats=len(src)), np.repeat(etype[-1], repeats=len(src))))  # 将起始节点ID和终点ID匹配
            self.positive_pairs.extend(pos_pars)  # self.positive_pairs:确定具有连接的节点对

        def generate_class_class_pairs(src, dst, src_type, positive_pairs):
            for src1, dst1, src_type1, dst_type1 in positive_pairs:
                if dst1 == dst and dst_type1 == 'class_node' and src_type1 == 'class_node' and src1 != src:
                    t = (src, src1, src_type, src_type1)
                    return t

        def generate_type_pairs(src, dst, src_type, positive_pairs):
            for src1, dst1, src_type1, dst_type1 in positive_pairs:
                if dst1 == dst and dst_type1 == 'type_node' and src1 != src:
                    t = (src, src1, src_type, src_type1)
                    return t

        def generate_attribute_value_pairs(src, dst, src_type, positive_pairs):
            for src1, dst1, src_type1, dst_type1 in positive_pairs:
                if dst1 == dst and dst_type1 == 'attribute_node' and src_type1 == 'value_node' and src1 != src:
                    t = (src, src1, src_type, src_type1)
                    return t

        # extended_pairs = set()
        # for src, dst, src_type, dst_type in self.positive_pairs:
        #     generated = None
        #     if (src_type, dst_type) == ('class_node', 'class_node'):
        #         generated = generate_class_class_pairs(src, dst, src_type, self.positive_pairs)
        #     elif src_type == 'type_node':
        #         generated = generate_type_pairs(src, dst, src_type, self.positive_pairs)
        #     elif (src_type, dst_type) == ('value_node', 'attribute_node'):
        #         generated = generate_attribute_value_pairs(src, dst, src_type, self.positive_pairs)
        #
        #     # 直接添加有效元素
        #     if generated and isinstance(generated, tuple) and len(generated) == 4:
        #         # 标准化排序并保留类型
        #         a, b = sorted(generated[:2])
        #         sorted_pair = (a, b) + generated[2:]
        #         extended_pairs.add(sorted_pair)
        #
        # self.positive_pairs.extend(extended_pairs)

        extended_pairs = set()
        for src, dst, src_type, dst_type in self.positive_pairs:
            if (src_type, dst_type) == ('class_node', 'class_node'):
                extended_pairs.add(generate_class_class_pairs(src, dst, src_type, self.positive_pairs))
            elif dst_type == 'type_node':
                extended_pairs.add(generate_type_pairs(src, dst, src_type, self.positive_pairs))
            elif (src_type, dst_type) == ('value_node', 'attribute_node'):
                extended_pairs.add(generate_attribute_value_pairs(src, dst, src_type, self.positive_pairs))

        filtered_set = set()
        for pair in extended_pairs:
            if pair is not None and isinstance(pair, tuple) and len(pair) == 4:
                filtered_set.add(pair)
        extended_pairs = filtered_set

        unique_pairs = set()
        # 用于存储要删除的元素
        to_remove = set()
        for pair in extended_pairs:
            # 将对进行排序
            sorted_pair = tuple(sorted(pair[:2])) + pair[2:]
            # 如果排序后的对已经在新集合中，则标记为要删除
            if sorted_pair in unique_pairs:
                to_remove.add(pair)
            else:
                # 否则，添加到新集合中
                unique_pairs.add(sorted_pair)
        # 从原始集合中删除重复元素
        extended_pairs -= to_remove
        self.positive_pairs.extend(extended_pairs)

        if seed is not None:
            np.random.seed(seed)
        for src, dst, src_type, dst_type in self.positive_pairs:
            neg_item = self.positive_pairs[np.random.choice(range(len(self.positive_pairs)))]
            # 随机挑选与当前节点类型不同，且不相连，且在negative中不重复的样本对，作为负样本
            while (neg_item[1] == dst or src_type == neg_item[-1] or (src, neg_item[1], src_type, neg_item[-1]) in self.negative_pairs or
                   self.has_edge_between_nodes(src, neg_item[1])):
                neg_item = self.positive_pairs[np.random.choice(range(len(self.positive_pairs)))]
            self.negative_pairs.append((src, neg_item[1], src_type, neg_item[-1]))

    def train(self, epoch, early_stopping, in_feat, temperature, max_norm, save_path):

        def _get_node_features(node_feature: dict, node_information, _in_feat):
            results = torch.zeros(size=(len(node_information), _in_feat))
            for idx, (node_id, node_type) in enumerate(node_information):
                results[idx] = node_feature[node_type][node_id]
            return results

        self.model.train()
        result = self.model(self.graph, self.graph.ndata['features'])

        anchor = _get_node_features(result, [(tup[0], tup[2]) for tup in self.positive_pairs], in_feat)
        positive = _get_node_features(result, [(tup[1], tup[-1]) for tup in self.positive_pairs], in_feat)
        negative = _get_node_features(result, [(tup[1], tup[-1]) for tup in self.negative_pairs], in_feat)

        # 规范化输入向量
        anchor = F.normalize(anchor, p=2, dim=1)
        positive = F.normalize(positive, p=2, dim=1)
        negative = F.normalize(negative, p=2, dim=1)

        # 检查数据中是否包含 nan 或 inf 值
        if torch.isnan(anchor).any() or torch.isnan(positive).any() or torch.isnan(negative).any():
            raise ValueError("Input contains NaN values.")

        if torch.isinf(anchor).any() or torch.isinf(positive).any() or torch.isinf(negative).any():
            raise ValueError("Input contains Inf values.")

        loss = info_nce_loss(anchor, positive, negative, temperature)
        print("[{}]: {}".format(epoch, loss))

        self.optimizer.zero_grad()
        loss.backward(retain_graph=True)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm)
        self.optimizer.step()

        new_node_features = self.model(self.graph, self.graph.ndata['features'])
        self.node_features = {k: v.clone().detach() for k, v in new_node_features.items()}

        wandb.log({
            "epoch": epoch,
            "train/loss": loss,
            "node_features": self.node_features
        })

        early_stopping(loss, self.model)  # 达到早停止条件时，early_stop会被置为True
        if loss <= self.best_loss:
            self.best_loss = loss
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            torch.save(self.model.state_dict(), os.path.join(save_path, str(self.edge_direction)+"_best_model.pth"))


class RGCNTrainer(object):
    def __init__(self, model, train_loader, optimizer, loss_fn):
        self.optimizer = optimizer
        self.model = model
        self.train_loader = train_loader
        self.loss_fn = loss_fn

    def train(self, epoch, device, batch_size):
        tqdm_batch = tqdm(self.train_loader, desc='Epoch-{} training'.format(epoch))
        self.model.train()
        train_loss = 0.0
        train_pred = []
        count = 0.0
        train_true = []

        for i, (batched_graph, labels) in enumerate(tqdm_batch):
            batched_graph = batched_graph.to(device)
            labels = labels.to(device)

            self.optimizer.zero_grad()
            outs = self.model(batched_graph)
            loss = self.loss_fn(outs, labels)
            loss.backward()
            self.optimizer.step()

            pred = outs.detach().max(dim=1)[1]
            train_loss += loss.item() * batch_size
            count += batch_size
            train_pred.append(pred.detach().cpu().numpy())
            train_true.append(labels.detach().cpu().numpy())

        train_true = np.concatenate(train_true)
        train_pred = np.concatenate(train_pred)

        # Calculate metrics
        avg_train_loss = train_loss * 1.0 / count
        train_overall_acc = metrics.accuracy_score(train_true, train_pred)
        train_balanced_acc = metrics.balanced_accuracy_score(train_true, train_pred)

        # Log metrics to wandb
        wandb.log({
            "epoch": epoch,
            "train/loss": avg_train_loss,
            "train/overall_acc": train_overall_acc,
            "train/balanced_acc": train_balanced_acc
        })

        print("Epoch{} - Train Loss:{}, Train Overall Accuracy:{}, Train Balanced Accuracy:{}".format(epoch, avg_train_loss, train_overall_acc, train_balanced_acc))
        torch.cuda.empty_cache()


class MVCNNTrainer(object):
    def __init__(self, model, train_loader, optimizer, loss_fn):
        self.optimizer = optimizer
        self.model = model
        self.train_loader = train_loader
        self.loss_fn = loss_fn

    def train(self, epoch, device, batch_size):
        tqdm_batch = tqdm(self.train_loader, desc='Epoch-{} training'.format(epoch))
        self.model.train()
        train_loss = 0.0
        train_pred = []
        count = 0.0
        train_true = []

        for i, (inputs, targets) in enumerate(tqdm_batch):
            # Convert from list of 3D to 4D
            inputs = np.stack(inputs, axis=1)
            inputs = torch.from_numpy(inputs)
            N, V, C, H, W = inputs.size()
            inputs = inputs.view(-1, C, H, W)
            inputs, targets = inputs.cuda(device), targets.cuda(device)

            # compute output
            out = self.model(inputs)
            loss = self.loss_fn(out, targets)

            # compute gradient and do SGD step
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            pred = out.max(dim=1)[1]
            train_loss += loss.item() * batch_size
            count += batch_size
            train_pred.append(pred.detach().cpu().numpy())
            train_true.append(targets.cpu().numpy())

        train_true = np.concatenate(train_true)
        train_pred = np.concatenate(train_pred)

        # Calculate metrics
        avg_train_loss = train_loss * 1.0 / count
        train_overall_acc = metrics.accuracy_score(train_true, train_pred)
        train_balanced_acc = metrics.balanced_accuracy_score(train_true, train_pred)

        # Log metrics to wandb
        wandb.log({
            "epoch": epoch,
            "train/loss": avg_train_loss,
            "train/overall_acc": train_overall_acc,
            "train/balanced_acc": train_balanced_acc
        })

        print("Epoch{} - Train Loss:{}, Train Overall Accuracy:{}, Train Balanced Accuracy:{}".format(epoch, avg_train_loss, train_overall_acc, train_balanced_acc))
