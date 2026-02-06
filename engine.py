import torch
from torch.autograd import Variable
from tensorboardX import SummaryWriter

from utils import *
from metrics import MetronAtK
import random
import copy
from data import UserItemRatingDataset
from torch.utils.data import DataLoader
import networkx as nx
from sklearn.preprocessing import normalize
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import cosine_similarity
import community.community_louvain as community_louvain
import torch.nn.functional as F
from collections import OrderedDict


class Engine(object):
    """Meta Engine for training & evaluating NCF model

    Note: Subclass should implement self.model !
    """

    def __init__(self, config):
        self.config = config  # model configuration
        self._metron = MetronAtK(top_k=10)
        # self._writer = SummaryWriter(log_dir='runs/{}'.format(config['alias']))  # tensorboard writer
        # self._writer.add_text('config', str(config), 0)
        self.server_model_param = {}
        self.client_model_params = {}
        self.reg = config['reg']
        self.num_user_clusters = config['num_user_clusters']
        self.num_item_clusters = config['num_item_clusters']
        self.resolution = config['resolution']
        self.top_k = config['top_k']
        self.labels = torch.tensor([1, 2])
        # explicit feedback
        # self.crit = torch.nn.MSELoss()
        # implicit feedback
        self.crit = torch.nn.BCELoss()
        self.groups = {}

    def instance_user_train_loader(self, user_train_data):
        """instance a user's train loader."""
        dataset = UserItemRatingDataset(user_tensor=torch.LongTensor(user_train_data[0]),
                                        item_tensor=torch.LongTensor(user_train_data[1]),
                                        target_tensor=torch.FloatTensor(user_train_data[2]))
        return DataLoader(dataset, batch_size=self.config['batch_size'], shuffle=True)

    def fed_train_single_batch(self, model_client, batch_data, optimizers, labels):
        """train a batch and return an updated model."""
        # load batch data.
        users, items, ratings = batch_data[0], batch_data[1], batch_data[2]
        ratings = ratings.float()

        if self.config['use_cuda'] is True:
            users, items, ratings = users.cuda(), items.cuda(), ratings.cuda()

        optimizer, optimizer_u, optimizer_i = optimizers

        # update mlp.
        optimizer.zero_grad()
        optimizer_u.zero_grad()
        optimizer_i.zero_grad()

        ratings_pred, supcon = model_client.forward_train(items, labels)
        # ratings_pred = model_client(items)
        loss = self.crit(ratings_pred.view(-1), ratings)
        loss = loss + self.reg * supcon

        loss.backward()
        # torch.nn.utils.clip_grad_norm_(model_client.parameters(), 3.0)

        optimizer.step()
        optimizer_u.step()
        optimizer_i.step()

        return model_client, loss.item()

    def aggregate_clients_params(self, round_user_params):
        """receive client models' parameters in a round, aggregate them and store the aggregated result for server."""
        # aggregate item embedding and score function via averaged aggregation.
        t = 0
        for user in round_user_params.keys():
            # load a user's parameters.
            user_params = round_user_params[user]
            # print(user_params)
            if t == 0:
                self.server_model_param = copy.deepcopy(user_params)
                for key in self.server_model_param.keys():
                    self.server_model_param[key].data = self.server_model_param[key].data / len(round_user_params)
            else:
                for key in user_params.keys():
                    self.server_model_param[key].data += user_params[key].data / len(round_user_params)
            t = 1

        for key in self.server_model_param.keys():
            for user in round_user_params.keys():
                self.client_model_params[user][key].data = copy.deepcopy(self.server_model_param[key].data)

    def compute_cluster(self, neighborhood_users, round_user_params):
        len_cluster = 0
        center_cluster = copy.deepcopy(round_user_params[0])
        for user_id in neighborhood_users:
            if len_cluster == 0:
                for key in center_cluster.keys():
                    center_cluster[key].data = round_user_params[user_id][key].data
                len_cluster += 1
            else:
                for key in center_cluster.keys():
                    center_cluster[key].data += round_user_params[user_id][key].data
                len_cluster += 1
        for key in center_cluster.keys():
            center_cluster[key].data = center_cluster[key].data / len_cluster
        return center_cluster

    def louvain_grouping(self, round_user_params, participants, indices):
        groups = {}

        # global_item_embedding = self.server_model_param['embedding_item.weight'].data[indices].view(-1)
        item_embedding = torch.stack([
            round_user_params[u]['embedding_item.weight'].data[indices].view(-1) #- global_item_embedding
            for u in participants
        ])

        dim = min(32, item_embedding.shape[1])
        pca = PCA(n_components=dim, random_state=42)
        item_embedding = pca.fit_transform(item_embedding.cpu().numpy())

        similarity_matrix = cosine_similarity(item_embedding)

        G = nx.Graph()

        added_edges = set()

        top_k = int(self.top_k * len(participants))

        for i, u1 in enumerate(participants):
            G.add_node(u1)
            sim_vector = similarity_matrix[i, :]

            topk_indices = np.argpartition(sim_vector, -top_k)[-top_k:]

            for j in topk_indices:
                u2 = participants[j]
                if i == j:
                    continue

                sim_val = similarity_matrix[i, j].item()
                if sim_val > 0:
                    u_min = min(u1, u2)
                    u_max = max(u1, u2)

                    if (u_min, u_max) not in added_edges:
                        G.add_edge(u1, u2, weight=sim_val)
                        added_edges.add((u_min, u_max))

        partition = community_louvain.best_partition(G, weight='weight', resolution=self.resolution)
        for user_id, community_id in partition.items():
            if community_id not in groups:
                groups[community_id] = []
            groups[community_id].append(user_id)

        return groups


    def fed_train_a_round(self, all_train_data, round_id):
        """train a round."""
        # sample users participating in single round.
        if self.config['clients_sample_ratio'] <= 1:
            num_participants = int(self.config['num_users'] * self.config['clients_sample_ratio'])
            participants = random.sample(range(self.config['num_users']), num_participants)
        else:
            participants = random.sample(range(self.config['num_users']), self.config['clients_sample_num'])

        # store users' model parameters of current round.
        round_participant_params = {}
        # store all the users' train loss and mae.
        all_loss = {}

        # perform model update for each participated user.
        for user in participants:
            loss = 0
            # copy the client model architecture from self.model
            model_client = copy.deepcopy(self.model)
            # for the first round, client models copy initialized parameters directly.
            # for other rounds, client models receive updated item embedding and score function from server.
            if round_id != 0:
                # user_param_dict = copy.deepcopy(self.model.state_dict())
                user_param_dict = {k: v.detach().clone() for k, v in self.model.state_dict().items()}
                if user in self.client_model_params.keys():
                    for key, value in self.client_model_params[user].items():
                        user_param_dict[key] = value.detach().clone()
                    model_client.load_state_dict(user_param_dict)
            # Defining optimizers
            # optimizer is responsible for updating score function.
            optimizer = torch.optim.SGD(model_client.mlp.parameters(),
                                        lr=self.config['lr'], weight_decay=self.config['l2_regularization'])  # MLP optimizer
            # optimizer_u is responsible for updating user embedding.
            optimizer_u = torch.optim.SGD(model_client.embedding_user.parameters(),
                                          lr=self.config['lr'] * self.config['lr_eta'],
                                          weight_decay=self.config['l2_regularization'])  # User optimizer
            # optimizer_i is responsible for updating item embedding.
            optimizer_i = torch.optim.SGD(model_client.embedding_item.parameters(),
                                          lr=self.config['lr'] * self.config['num_items'] * self.config['lr_eta'],
                                          weight_decay=self.config['l2_regularization'])  # Item optimizer
            optimizers = [optimizer, optimizer_u, optimizer_i]

            # load current user's training data and instance a train loader.
            user_train_data = [all_train_data[0][user], all_train_data[1][user], all_train_data[2][user]]
            user_dataloader = self.instance_user_train_loader(user_train_data)
            model_client.train()
            sample_num = 0
            # update client model.
            for epoch in range(self.config['local_epoch']):
                for batch_id, batch in enumerate(user_dataloader):
                    assert isinstance(batch[0], torch.LongTensor)
                    model_client, loss_u = self.fed_train_single_batch(model_client, batch, optimizers, self.labels)
                    loss += loss_u * len(batch[0])
                    sample_num += len(batch[0])
                all_loss[user] = loss / sample_num
            # obtain client model parameters.
            client_param = model_client.state_dict()
            # store client models' local parameters for personalization.
            self.client_model_params[user] = {k: v.detach().cpu().clone() for k, v in client_param.items()}
            round_participant_params[user] = {k: v.clone() for k, v in self.client_model_params[user].items()}
            del round_participant_params[user]['embedding_user.weight']

            for key in ["mlp.4.weight", "mlp.4.bias"]:
                del round_participant_params[user][key]

        # aggregate client models in server side.
        self.aggregate_clients_params(round_participant_params)
        kmeans = KMeans(n_clusters=self.num_item_clusters)
        item_clusters = kmeans.fit_predict(self.server_model_param['embedding_item.weight'].data)
        self.labels = copy.deepcopy(torch.tensor(item_clusters, dtype=torch.long))

        selected_item_cluster = random.choice(range(self.num_item_clusters))
        indices = [i for i, x in enumerate(item_clusters) if x == selected_item_cluster]

        self.groups = self.louvain_grouping(round_participant_params, participants, indices)
        print(f"groups num: {len(self.groups.keys())}")
        print(sorted([len(value) for value in self.groups.values()], reverse=True))

        for group_id, group in self.groups.items():
            cluster_center = self.compute_cluster(group, round_participant_params)
            for key, param in cluster_center.items():
                param = param.detach().clone()#.cuda()
                for i, user_id in enumerate(group):
                    self.client_model_params[user_id][key].copy_(param)

        return all_loss, len(self.groups)

    def fed_evaluate(self, evaluate_data):
        neg_sample = 99
        # evaluate all client models' performance using testing data.
        test_users, test_items = evaluate_data[0], evaluate_data[1]
        negative_users, negative_items = evaluate_data[2], evaluate_data[3]
        # ratings for computing loss.
        temp = [0] * (neg_sample + 1)
        temp[0] = 1
        ratings = torch.FloatTensor(temp)
        if self.config['use_cuda'] is True:
            test_users = test_users.cuda()
            test_items = test_items.cuda()
            negative_users = negative_users.cuda()
            negative_items = negative_items.cuda()
            ratings = ratings.cuda()
        # store all users' test item prediction score.
        test_scores = None
        # store all users' negative items prediction scores.
        negative_scores = None
        all_loss = {}
        for user in range(self.config['num_users']):
            # load each user's mlp parameters.
            user_model = copy.deepcopy(self.model)
            if user in self.client_model_params.keys():
                user_param_dict = copy.deepcopy(self.client_model_params[user])
                for key in user_param_dict.keys():
                    user_param_dict[key] = user_param_dict[key].data.cuda()
            else:
                user_param_dict = copy.deepcopy(self.model.state_dict())
            user_model.load_state_dict(user_param_dict)
            user_model.eval()
            with torch.no_grad():
                # obtain user's positive test information.
                test_user = test_users[user: user + 1]
                test_item = test_items[user: user + 1]
                # obtain user's negative test information.
                negative_user = negative_users[user * neg_sample: (user + 1) * neg_sample]
                negative_item = negative_items[user * neg_sample: (user + 1) * neg_sample]
                # perform model prediction.
                test_score = user_model(test_item)
                negative_score = user_model(negative_item)
                if user == 0:
                    test_scores = test_score
                    negative_scores = negative_score
                else:
                    test_scores = torch.cat((test_scores, test_score))
                    negative_scores = torch.cat((negative_scores, negative_score))
                ratings_pred = torch.cat((test_score, negative_score))
                loss = self.crit(ratings_pred.view(-1), ratings)
            all_loss[user] = loss.item()
        if self.config['use_cuda'] is True:
            test_users = test_users.cpu()
            test_items = test_items.cpu()
            test_scores = test_scores.cpu()
            negative_users = negative_users.cpu()
            negative_items = negative_items.cpu()
            negative_scores = negative_scores.cpu()
        self._metron.subjects = [test_users.data.view(-1).tolist(),
                                 test_items.data.view(-1).tolist(),
                                 test_scores.data.view(-1).tolist(),
                                 negative_users.data.view(-1).tolist(),
                                 negative_items.data.view(-1).tolist(),
                                 negative_scores.data.view(-1).tolist()]
        hit_ratio_5, ndcg_5 = self._metron.cal_hit_ratio(5), self._metron.cal_ndcg(5)
        hit_ratio_10, ndcg_10 = self._metron.cal_hit_ratio(10), self._metron.cal_ndcg(10)
        hit_ratio_20, ndcg_20 = self._metron.cal_hit_ratio(20), self._metron.cal_ndcg(20)
        return {5: hit_ratio_5, 10: hit_ratio_10, 20: hit_ratio_20}, {5: ndcg_5, 10: ndcg_10, 20: ndcg_20}, all_loss

    def save(self, alias, epoch_id, hit_ratio, ndcg):
        assert hasattr(self, 'model'), 'Please specify the exact model !'
        model_dir = self.config['model_dir'].format(alias, epoch_id, hit_ratio, ndcg)
        save_checkpoint(self.model, model_dir)
