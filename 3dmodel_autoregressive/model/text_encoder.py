from .layers.transformer import *
from .layers.improved_transformer import *
from .layers.positional_encoding import *
from .model_utils import _make_seq_first, _make_batch_first, \
    _get_padding_mask, _get_key_padding_mask, _get_group_mask


class CADEmbedding(nn.Module):
    """Embedding: positional embed + command embed + parameter embed + group embed (optional)"""
    def __init__(self, cfg, seq_len, use_group=False, group_len=None):
        super().__init__()

        self.command_embed = nn.Embedding(cfg.n_commands, cfg.d_model)

        args_dim = cfg.args_dim + 1
        self.arg_embed = nn.Embedding(args_dim, 64, padding_idx=0)
        self.embed_fcn = nn.Linear(64 * cfg.n_args, cfg.d_model)

        # use_group: additional embedding for each sketch-extrusion pair
        self.use_group = use_group
        if use_group:
            if group_len is None:
                group_len = cfg.max_num_groups
            self.group_embed = nn.Embedding(group_len + 2, cfg.d_model)

        self.pos_encoding = PositionalEncodingLUT(cfg.d_model, max_len=seq_len+2)

    def forward(self, commands, args, groups=None):
        #print('commands.shape: ',commands.shape)
        '''commands.shape:  torch.Size([60, 10])'''
        S, N = commands.shape

        src = self.command_embed(commands.long()) + \
              self.embed_fcn(self.arg_embed((args + 1).long()).view(S, N, -1))  # shift due to -1 PAD_VAL

        if self.use_group:
            src = src + self.group_embed(groups.long())

        src = self.pos_encoding(src)

        return src


class Encoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        seq_len = cfg.max_total_len
        self.use_group = cfg.use_group_emb
        self.embedding = CADEmbedding(cfg, seq_len, use_group=self.use_group)

        encoder_layer = TransformerEncoderLayerImproved(cfg.d_model, cfg.n_heads, cfg.dim_feedforward, cfg.dropout)
        encoder_norm = LayerNorm(cfg.d_model)
        self.encoder = TransformerEncoder(encoder_layer, cfg.n_layers, encoder_norm)

    def forward(self, commands, args):
        # print('commands.shape: ',commands.shape)
        # print('args.shape: ',args.shape)
        '''commands.shape:  torch.Size([60, 10])
            args.shape:  torch.Size([60, 10, 16])'''
        padding_mask, key_padding_mask = _get_padding_mask(commands, seq_dim=0), _get_key_padding_mask(commands, seq_dim=0)
        #print('padding_mask: ',padding_mask)
        #print('key_padding_mask: ',key_padding_mask)
        # print('padding_mask.shape: ',padding_mask.shape)
        # print('key_padding_mask.shape: ',key_padding_mask.shape)
        '''
            padding_mask.shape:  torch.Size([60, 10, 1])
            key_padding_mask.shape:  torch.Size([10, 60])
        '''
        
        group_mask = _get_group_mask(commands, seq_dim=0) if self.use_group else None
        src = self.embedding(commands, args, group_mask)
        memory = self.encoder(src, mask=None, src_key_padding_mask=key_padding_mask)
        # print('group_mask.shape: ',group_mask.shape)
        # print('src.shape: ',src.shape)
        # print('memory.shape: ',memory.shape)
        '''
            group_mask.shape:  torch.Size([60, 10])
            src.shape:  torch.Size([60, 10, 256])
            memory.shape:  torch.Size([60, 10, 256])
        '''
        z = (memory * padding_mask) / padding_mask.sum(dim=0, keepdim=True) 
        return z



class Bottleneck(nn.Module):
    def __init__(self, cfg):
        super(Bottleneck, self).__init__()

        self.bottleneck = nn.Sequential(nn.Linear(cfg.d_model, cfg.dim_z),
                                        nn.Tanh())

    def forward(self, z):
        return self.bottleneck(z)


class Text_Encoder(nn.Module):
    def __init__(self, cfg):
        super(CADTransformer, self).__init__()

        self.args_dim = cfg.args_dim + 1

        self.encoder = Encoder(cfg)

        self.bottleneck = Bottleneck(cfg)

    def forward(self, commands_enc, args_enc,
                z=None, return_tgt=True, encode_mode=False):
        commands_enc_, args_enc_ = _make_seq_first(commands_enc, args_enc)  # Possibly None, None
        # print('commands_enc_.shape: ',commands_enc_.shape)
        # print('args_enc_.shape: ',args_enc_.shape)
        '''commands_enc_.shape:  torch.Size([60, 10])
            args_enc_.shape:  torch.Size([60, 10, 16])'''
        z = self.encoder(commands_enc_, args_enc_)
        z = self.bottleneck(z)
        # torch.Size([60, 10, 16])

        return z
