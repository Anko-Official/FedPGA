import torch
import torch.nn as nn
from engine import Engine


class MLP(torch.nn.Module):
    def __init__(self, config):
        super(MLP, self).__init__()
        self.config = config
        self.num_users = config['num_users']
        self.num_items = config['num_items']
        self.latent_dim = config['latent_dim']
        self.temperature = 0.3
        self.base_temperature = 0.5

        self.embedding_user = nn.Embedding(num_embeddings=1, embedding_dim=self.latent_dim)
        self.embedding_item = nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.latent_dim)

        self.mlp = nn.Sequential(
            nn.Linear(self.latent_dim * 2, self.latent_dim),
            nn.ReLU(),
            # nn.Dropout(p=0.1),
            nn.Linear(self.latent_dim, self.latent_dim // 2),
            nn.ReLU(),
            # nn.Dropout(p=0.2),
            nn.Linear(self.latent_dim // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, item_indices):
        user_embedding = self.embedding_user(torch.tensor([0] * len(item_indices)).cuda())
        item_embedding = self.embedding_item(item_indices)
        return self.mlp(torch.cat([user_embedding, item_embedding], dim=-1))

    def forward_train(self, item_indices, labels):
        user_embedding = self.embedding_user(torch.tensor([0] * len(item_indices)).cuda())
        item_embedding = self.embedding_item(item_indices)
        rating = self.mlp(torch.cat([user_embedding, item_embedding], dim=-1))

        if labels.shape[0] == 2:
            supcon = torch.tensor(0)
        else:
            embedding = torch.nn.functional.normalize(self.embedding_item.weight)
            supcon = self.supcon(embedding, labels)

        return rating, supcon

    def supcon(self, features, labels=None, mask=None):
        """Compute loss for model.
        Args:
            features: hidden vector of shape [bsz, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz].
        Returns:
            A loss scalar.
        """
        device = features.device

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        # compute logits
        anchor_dot_contrast = torch.div(torch.matmul(features, features.T), self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask
        exp_logits = torch.exp(logits) * logits_mask

        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-6)
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-6)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.mean()

        return loss

    def init_weight(self):
        pass

    def load_pretrain_weights(self):
        pass


class MLPEngine(Engine):
    """Engine for training & evaluating GMF model"""
    def __init__(self, config):
        self.model = MLP(config)
        if config['use_cuda'] is True:
            # use_cuda(True, config['device_id'])
            self.model.cuda()
        super(MLPEngine, self).__init__(config)
        print(self.model)
