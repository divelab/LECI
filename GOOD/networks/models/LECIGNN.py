r"""
`Joint Learning of Label and Environment Causal Independence for Graph Out-of-Distribution Generalization <https://arxiv.org/abs/2306.01103>`_.
"""
import munch
import torch
import torch.nn as nn
from torch import Tensor
from torch.autograd import Function
from torch_geometric.nn import InstanceNorm
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import is_undirected
from torch_sparse import transpose

from GOOD import register
from GOOD.utils.config_reader import Union, CommonArgs, Munch
from .BaseGNN import GNNBasic
from .Classifiers import Classifier
from .GINs import GINFeatExtractor
from .GINvirtualnode import vGINFeatExtractor
from .Pooling import GlobalMeanPool
from munch import munchify
from .MolEncoders import AtomEncoder, BondEncoder
from GOOD.utils.fast_pytorch_kmeans import KMeans


@register.model_register
class LECIGIN(GNNBasic):

    def __init__(self, config: Union[CommonArgs, Munch]):
        super(LECIGIN, self).__init__(config)

        # --- if environment inference ---
        config.environment_inference = False
        self.env_infer_warning = f'#W#Expermental mode: environment inference phase.'
        config.dataset.num_envs = 3
        # --- Test environment inference ---

        self.config = config

        self.learn_edge_att = True
        self.LA = config.ood.extra_param[0]
        self.EC = config.ood.extra_param[1]      # Never used
        self.EA = config.ood.extra_param[2]
        self.EF = config.ood.extra_param[4]

        fe_kwargs = {'without_embed': True if self.EF else False}

        # --- Build networks ---
        self.sub_gnn = GINFeatExtractor(config, **fe_kwargs)
        self.extractor = ExtractorMLP(config)

        self.ef_mlp = EFMLP(config, bn=True)
        self.ef_discr_mlp = MLP([config.model.dim_hidden, 2 * config.model.dim_hidden, config.model.dim_hidden],
                                 dropout=config.model.dropout_rate, config=config, bn=True)
        self.ef_pool = GlobalMeanPool()
        self.ef_classifier = Classifier(munchify({'model': {'dim_hidden': config.model.dim_hidden},
                                                   'dataset': {'num_classes': config.dataset.num_envs}}))



        self.lc_gnn = GINFeatExtractor(config, **fe_kwargs)
        self.la_gnn = GINFeatExtractor(config, **fe_kwargs)
        self.ec_gnn = GINFeatExtractor(config, **fe_kwargs)    # Never used
        self.ea_gnn = GINFeatExtractor(config, **fe_kwargs)

        self.lc_classifier = Classifier(config)
        self.la_classifier = Classifier(config)
        self.ec_classifier = Classifier(munchify({'model': {'dim_hidden': config.model.dim_hidden},
                                                   'dataset': {'num_classes': config.dataset.num_envs}})) # Never used
        self.ea_classifier = Classifier(munchify({'model': {'dim_hidden': config.model.dim_hidden},
                                                  'dataset': {'num_classes': config.dataset.num_envs}}))

        self.edge_mask = None



    def forward(self, *args, **kwargs):
        r"""
        The LECIGIN model implementation.

        Args:
            *args (list): argument list for the use of arguments_read. Refer to :func:`arguments_read <GOOD.networks.models.BaseGNN.GNNBasic.arguments_read>`
            **kwargs (dict): key word arguments for the use of arguments_read. Refer to :func:`arguments_read <GOOD.networks.models.BaseGNN.GNNBasic.arguments_read>`

        Returns (Tensor):
            Label predictions and other results for loss calculations.

        """
        data = kwargs.get('data')

        # --- Filter environment info in features (only features) ---
        if self.EF:
            filtered_features = self.ef_mlp(data.x, data.batch)
            adversarial_features = GradientReverseLayerF.apply(filtered_features, self.EF * self.config.train.alpha)
            ef_logits = self.ef_classifier(self.ef_pool(self.ef_discr_mlp(adversarial_features, data.batch), data.batch))
            data.x = filtered_features
            kwargs['data'] = data
        else:
            ef_logits = None


        node_repr = self.sub_gnn.get_node_repr(*args, **kwargs)
        att_log_logits = self.extractor(node_repr, data.edge_index, data.batch)
        att = self.sampling(att_log_logits, self.training)

        if self.learn_edge_att:
            if is_undirected(data.edge_index):
                nodesize = data.x.shape[0]
                edge_att = (att + transpose(data.edge_index, att, nodesize, nodesize, coalesced=False)[1]) / 2
            else:
                edge_att = att
        else:
            edge_att = self.lift_node_att_to_edge_att(att, data.edge_index)


        set_masks(edge_att, self.lc_gnn)
        lc_logits = self.lc_classifier(self.lc_gnn(*args, **kwargs))
        clear_masks(self)

        if self.LA and self.training:
            set_masks(1 - GradientReverseLayerF.apply(edge_att, self.LA * self.config.train.alpha), self.la_gnn)
            la_logits = self.la_classifier(self.la_gnn(*args, **kwargs))
            clear_masks(self)
        else:
            la_logits = None

        if self.EA and self.training:
            set_masks(GradientReverseLayerF.apply(edge_att, self.EA * self.config.train.alpha), self.ea_gnn)
            ea_readout = self.ea_gnn(*args, **kwargs)
            if self.config.environment_inference:
                if self.env_infer_warning:
                    print(self.env_infer_warning)
                    self.env_infer_warning = None
                kmeans = KMeans(n_clusters=self.config.dataset.num_envs, n_init=10, device=ea_readout.device).fit(ea_readout)
                self.E_infer = kmeans.labels_
            ea_logits = self.ea_classifier(ea_readout)
            clear_masks(self)
        else:
            ea_logits = None

        self.edge_mask = edge_att

        return (lc_logits, la_logits, None, ea_logits, ef_logits), att, edge_att

    def sampling(self, att_log_logits, training):
        temp = (self.config.train.epoch * 0.1 + (200 - self.config.train.epoch) * 10) / 200 if self.config.dataset.dataset_name == 'GOODMotif' and self.config.dataset.domain == 'size' else 1
        # temp = 1
        att = self.concrete_sample(att_log_logits, temp=temp, training=training)
        return att

    @staticmethod
    def lift_node_att_to_edge_att(node_att, edge_index):
        src_lifted_att = node_att[edge_index[0]]
        dst_lifted_att = node_att[edge_index[1]]
        edge_att = src_lifted_att * dst_lifted_att
        return edge_att

    @staticmethod
    def concrete_sample(att_log_logit, temp, training):
        if training:
            random_noise = torch.empty_like(att_log_logit).uniform_(1e-10, 1 - 1e-10)
            random_noise = torch.log(random_noise) - torch.log(1.0 - random_noise)
            att_bern = ((att_log_logit + random_noise) / temp).sigmoid()
        else:
            att_bern = (att_log_logit).sigmoid()
        return att_bern


@register.model_register
class LECIvGIN(LECIGIN):
    r"""
    The GIN virtual node version of LECI.
    """

    def __init__(self, config: Union[CommonArgs, Munch]):
        super(LECIvGIN, self).__init__(config)
        fe_kwargs = {'without_embed': True if self.EF else False}
        self.sub_gnn = vGINFeatExtractor(config, **fe_kwargs)
        self.lc_gnn = vGINFeatExtractor(config, **fe_kwargs)
        self.la_gnn = vGINFeatExtractor(config, **fe_kwargs)
        self.ec_gnn = vGINFeatExtractor(config, **fe_kwargs) # Never used
        self.ea_gnn = vGINFeatExtractor(config, **fe_kwargs)


class EFMLP(nn.Module):

    def __init__(self, config: Union[CommonArgs, Munch], bn):
        super(EFMLP, self).__init__()
        if config.dataset.dataset_type == 'mol':
            self.atom_encoder = AtomEncoder(config.model.dim_hidden, config)
            self.mlp = MLP([config.model.dim_hidden, config.model.dim_hidden, 2 * config.model.dim_hidden,
                            config.model.dim_hidden], config.model.dropout_rate, config, bn=bn)
        else:
            self.atom_encoder = nn.Identity()
            self.mlp = MLP([config.dataset.dim_node, config.model.dim_hidden, 2 * config.model.dim_hidden,
                            config.model.dim_hidden], config.model.dropout_rate, config, bn=bn)

    def forward(self, x, batch):
        return self.mlp(self.atom_encoder(x), batch)


class ExtractorMLP(nn.Module):

    def __init__(self, config: Union[CommonArgs, Munch]):
        super().__init__()
        hidden_size = config.model.dim_hidden
        self.learn_edge_att = True
        dropout_p = config.model.dropout_rate

        if self.learn_edge_att:
            self.feature_extractor = MLP([hidden_size * 2, hidden_size * 4, hidden_size, 1], dropout=dropout_p,
                                         config=config, bn=config.ood.extra_param[5])
        else:
            self.feature_extractor = MLP([hidden_size * 1, hidden_size * 2, hidden_size, 1], dropout=dropout_p,
                                         config=config, bn=config.ood.extra_param[5])

    def forward(self, emb, edge_index, batch):
        if self.learn_edge_att:
            col, row = edge_index
            f1, f2 = emb[col], emb[row]
            f12 = torch.cat([f1, f2], dim=-1)
            att_log_logits = self.feature_extractor(f12, batch[col])
        else:
            att_log_logits = self.feature_extractor(emb, batch)
        return att_log_logits


class BatchSequential(nn.Sequential):
    def forward(self, inputs, batch=None):
        for module in self._modules.values():
            if isinstance(module, (InstanceNorm)):
                assert batch is not None
                inputs = module(inputs, batch)
            else:
                inputs = module(inputs)
        return inputs


class MLP(BatchSequential):
    def __init__(self, channels, dropout, config, bias=True, bn=False):
        m = []
        for i in range(1, len(channels)):
            m.append(nn.Linear(channels[i - 1], channels[i], bias))

            if i < len(channels) - 1:
                if bn:
                    m.append(nn.BatchNorm1d(channels[i]))
                else:
                    m.append(InstanceNorm(channels[i]))

                m.append(nn.ReLU())
                m.append(nn.Dropout(dropout))

        super(MLP, self).__init__(*m)

class GradientReverseLayerF(Function):
    r"""
    Gradient reverse layer for DANN algorithm.
    """
    @staticmethod
    def forward(ctx, x, alpha):
        r"""
        gradient forward propagation

        Args:
            ctx (object): object of the GradientReverseLayerF class
            x (Tensor): feature representations
            alpha (float): the GRL learning rate

        Returns (Tensor):
            feature representations

        """
        ctx.alpha = alpha
        return x.view_as(x)  # * alpha

    @staticmethod
    def backward(ctx, grad_output):
        r"""
        gradient backpropagation step

        Args:
            ctx (object): object of the GradientReverseLayerF class
            grad_output (Tensor): raw backpropagation gradient

        Returns (Tensor):
            backpropagation gradient

        """
        output = grad_output.neg() * ctx.alpha
        return output, None


def set_masks(mask: Tensor, model: nn.Module):
    r"""
    Modified from https://github.com/wuyxin/dir-gnn.
    """
    for module in model.modules():
        if isinstance(module, MessagePassing):
            module._apply_sigmoid = False
            module.__explain__ = True
            module._explain = True
            module.__edge_mask__ = mask
            module._edge_mask = mask


def clear_masks(model: nn.Module):
    r"""
    Modified from https://github.com/wuyxin/dir-gnn.
    """
    for module in model.modules():
        if isinstance(module, MessagePassing):
            module.__explain__ = False
            module._explain = False
            module.__edge_mask__ = None
            module._edge_mask = None
