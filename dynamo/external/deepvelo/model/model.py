from dgl import DGLGraph
import dgl.function as fn
import dgl
from .layers import GINLayer, ApplyNodeFunc, MLP
from dgl.nn.pytorch import GraphConv
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..base import BaseModel


class MnistModel(BaseModel):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, num_classes)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


class VeloModel(BaseModel):
    def __init__(self, n_genes, layers=[256, 64]):
        super().__init__()
        self.fc1 = nn.Linear(2 * n_genes, layers[0])
        self.fc2 = nn.Linear(layers[0], layers[1])
        self.fc3 = nn.Linear(layers[1], 2 * n_genes)

    def forward(self, x_u, x_s):
        """
        right now it is jus mlp, and the complexity of the middle part does not make sense;
        Change it to the attention model and constrain the information flow
        """
        batch, n_gene = x_u.shape
        # x should be (batch, features=2*n_gene)
        x = torch.cat([x_u, x_s], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))  # (batch, 64)
        x = self.fc3(x)  # (batch, genes*2)
        beta = x[:, 0:n_gene]  # (batch, genes)
        gamma = x[:, n_gene : 2 * n_gene]
        pred = beta * x_u + gamma * x_s
        return pred


class VeloTransformer(BaseModel):
    def __init__(self, n_genes, layers=[2, 64], emb_dim=32):
        super().__init__()
        self.emb = nn.Embedding(num_embeddings=2 * n_genes, embedding_dim=emb_dim)
        self.attn = nn.MultiheadAttention(embed_dim=emb_dim, num_heads=1)
        self.fc1 = nn.Linear(emb_dim, emb_dim)
        self.fc2 = nn.Linear(emb_dim, layers[0])
        self.fc3 = nn.Linear(layers[0] * 2 * n_genes, 2 * n_genes)

    def forward(self, x_u, x_s):
        """
        right now it is jus mlp, and the complexity of the middle part does not make sense;
        Change it to the attention model and constrain the information flow
        """
        batch, n_gene = x_u.shape
        # x should be (batch, features=2*n_gene)
        x = torch.cat([x_u, x_s], dim=1)
        x = x.unsqueeze(2)  # (batch, 2*genes, 1)
        x_emb = x * self.emb.weight  # (batch, 2*genes, emb_dim=32)
        x_emb = F.relu(self.fc1(x_emb))
        x_emb = x_emb.permute(1, 0, 2)  # (2*genes, batch, emb_dim)

        x, attn_output_weights = self.attn(
            x_emb, x_emb, x_emb
        )  # (2*genes, batch, emb_dim)
        # TODO: try an additional activation on the attention
        x = x.permute(1, 0, 2)  # (batch, 2*genes, emb_dim=32)
        x = F.relu(self.fc2(x)).reshape([batch, -1])  # (batch, 2*genes*d)
        x = self.fc3(x)  # (batch, genes*2)
        beta = x[:, 0:n_gene]  # (batch, genes)
        gamma = x[:, n_gene : 2 * n_gene]
        pred = beta * x_u + gamma * x_s
        return pred


class VeloGCN(BaseModel):
    def __init__(
        self,
        g,
        n_genes,
        layers=[64],
        fc_layer=False,
        pred_unspliced=False,
        dropout=0.2,
        activation=F.relu,
    ):
        super(VeloGCN, self).__init__()
        self.g = g  # the graph
        self.fc_layer = fc_layer
        self.pred_unspliced = pred_unspliced
        self.curent_kinetic_rates = {}

        self.layers = nn.ModuleList()
        # input layer
        self.layers.append(GraphConv(n_genes * 2, layers[0], activation=activation))
        # hidden layers
        for i in range(len(layers) - 1):
            self.layers.append(
                GraphConv(layers[i], layers[i + 1], activation=activation)
            )
        self.dropout = nn.Dropout(dropout)
        # output layer: which maps the embedded cell states tothe prediction of
        # kinetic rates for each gene. The intuition is we should know the cell's
        # developing pattern given the profile of current cell.
        if pred_unspliced:
            out_layer_dim = n_genes * 3
        else:
            out_layer_dim = n_genes * 2
        if fc_layer:
            self.layers.append(GraphConv(layers[-1], layers[-1]))
            self.fc = nn.Linear(layers[-1], out_layer_dim)
        else:
            self.layers.append(GraphConv(layers[-1], out_layer_dim))

    def to(self, device):
        """
        Move the model and graph to the specified device.
        """
        super().to(device)
        self.g = self.g.to(device)
        return self

    def forward(self, x_u, x_s):
        batch, n_gene = x_u.shape
        # h should be (batch, features=2*n_gene)
        h = torch.cat([x_u, x_s], dim=1)  # features
        for i, layer in enumerate(self.layers):
            h = layer(self.g, h)
            h = self.dropout(h)
        x = h

        # x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))  # (batch, 64)
        # x = self.fc3(x)  # (batch, genes*2)
        if self.fc_layer:
            x = self.fc(x)

        beta = x[:, 0:n_gene]  # (batch, genes)
        gamma = x[:, n_gene : 2 * n_gene]
        self.curent_kinetic_rates["beta"] = beta
        self.curent_kinetic_rates["gamma"] = gamma
        # if pred_unspliced, x is (batch, 3*n_genes)
        if self.pred_unspliced:
            pred = beta * x_u - gamma * x_s
            alpha = x[:, 2 * n_gene : 3 * n_gene]
            self.curent_kinetic_rates["alpha"] = alpha
            pred_u = alpha - beta * x_u
            pred = torch.cat([pred, pred_u], dim=1)
        else:
            pred = beta * x_u + gamma * x_s
        return pred

    def get_current_batch_kinetic_rates(self):
        return self.curent_kinetic_rates


# implement the model using GIN layers
class VeloGIN(BaseModel):
    def __init__(
        self,
        g,
        n_genes,
        layers=[64],
        n_mlp_layers=2,
        hidden_dim=64,
        res_connect=False,
        activation=F.relu,
    ):
        super(VeloGIN, self).__init__()
        self.g = g  # the graph
        self.layers = nn.ModuleList()
        # TODO(Haotian): test an embedding layer before the input
        # input layer
        mlp = MLP(n_mlp_layers, n_genes * 2, hidden_dim, layers[0])
        self.layers.append(
            GINLayer(
                ApplyNodeFunc(mlp),
                aggr_type="sum",
                dropout=0.0,
                batch_norm=True,
                residual=res_connect,
                init_eps=0,
                learn_eps=True,
            )
        )
        # hidden layers
        for i in range(len(layers) - 1):
            mlp = MLP(n_mlp_layers, layers[i], hidden_dim, layers[i + 1])
            self.layers.append(
                GINLayer(
                    ApplyNodeFunc(mlp),
                    aggr_type="sum",
                    dropout=0.0,
                    batch_norm=True,
                    residual=res_connect,
                    init_eps=0,
                    learn_eps=True,
                )
            )
        # output layer
        mlp = MLP(n_mlp_layers, layers[-1], hidden_dim, n_genes * 2)
        self.layers.append(
            GINLayer(
                ApplyNodeFunc(mlp),
                aggr_type="sum",
                dropout=0.0,
                batch_norm=True,
                residual=res_connect,
                init_eps=0,
                learn_eps=True,
            )
        )

    def forward(self, x_u, x_s):
        """
        right now it is jus mlp, and the complexity of the middle part does not make sense;
        Change it to the attention model and constrain the information flow
        """
        batch, n_gene = x_u.shape
        # h should be (batch, features=2*n_gene)
        h = torch.cat([x_u, x_s], dim=1)  # features
        for i, layer in enumerate(self.layers):
            h = layer(self.g, h)
        x = h

        # x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))  # (batch, 64)
        # x = self.fc3(x)  # (batch, genes*2)
        beta = x[:, 0:n_gene]  # (batch, genes)
        gamma = x[:, n_gene : 2 * n_gene]
        pred = beta * x_u + gamma * x_s
        return pred


# implement mini-batch using node-flow in DGL


class NodeUpdate(nn.Module):
    def __init__(self, in_feats, out_feats, activation=None, test=False, concat=False):
        super(NodeUpdate, self).__init__()
        self.linear = nn.Linear(in_feats, out_feats)
        self.activation = activation
        self.concat = concat
        self.test = test

    def forward(self, node):
        h = node.data["h"]
        if self.test:
            h = h * node.data["norm"]
        h = self.linear(h)
        # skip connection
        if self.concat:
            h = torch.cat((h, self.activation(h)), dim=1)
        elif self.activation:
            h = self.activation(h)
        return {"activation": h}


class VeloGCNNodeFlow(BaseModel):
    def __init__(self, n_genes, layers=[64], activation=F.relu, dropout=0, **kargs):
        super(VeloGCNNodeFlow, self).__init__()
        self.n_layers = len(layers)
        if dropout != 0:
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.dropout = None
        self.layers = nn.ModuleList()
        # input layer
        self.layers.append(NodeUpdate(n_genes * 2, layers[0], activation, concat=False))
        # hidden layers
        for i in range(len(layers) - 1):
            self.layers.append(
                NodeUpdate(layers[i], layers[i + 1], activation, concat=False)
            )
        # output layer
        self.layers.append(NodeUpdate(layers[-1], n_genes * 2))

    def forward(self, nf):
        assert nf.layers._graph.num_layers == len(self.layers) + 1
        x_u = nf.layers[0].data["Ux_sz"]
        x_s = nf.layers[0].data["Sx_sz"]
        batch, n_gene = x_u.shape
        input_activation = torch.cat([x_u, x_s], dim=1)
        # gnn propagate and output x (batch, feature_size)
        x = self._graph_forward(nf, input_activation)

        beta = x[:, 0:n_gene]  # (batch, genes)
        gamma = x[:, n_gene : 2 * n_gene]
        x_u_out, x_s_out = nf.layers[-1].data["Ux_sz"], nf.layers[-1].data["Sx_sz"]
        pred = beta * x_u_out + gamma * x_s_out
        return pred

    def _graph_forward(self, nf, input_activation):
        nf.layers[0].data["activation"] = input_activation

        for i, layer in enumerate(self.layers):
            h = nf.layers[i].data.pop("activation")
            if self.dropout:
                h = self.dropout(h)
            nf.layers[i].data["h"] = h
            nf.block_compute(
                i,
                fn.copy_src(src="h", out="m"),
                lambda node: {"h": node.mailbox["m"].mean(dim=1)},
                layer,
            )

        h = nf.layers[-1].data.pop("activation")
        return h

    # Implement the Pytorch Geometric version here
