import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import PastDecomposableMixing
from .autoformer import SeriesDecompositionBlock  # 如果需要，请将其包含在项目中
from .revin import RevIN  # 如果需要，请将其包含在项目中
from .transformer.embedding import DataEmbedding  # 如果需要，请将其包含在项目中


class BackboneTimeMixer(nn.Module):
    def __init__(
        self,
        task_name,
        n_steps,
        n_features,
        n_classes,
        n_layers=3,
        d_model=64,
        d_ffn=128,
        dropout=0.1,
        channel_independence=False,
        decomp_method='moving_avg',
        top_k=5,
        moving_avg=25,
        downsampling_layers=0,
        downsampling_window=2,
        downsampling_method='max',
        use_future_temporal_feature=False,
        embed="fixed",
        freq="h",
    ):
        super().__init__()
        self.task_name = task_name
        self.n_steps = n_steps
        self.n_features = n_features
        self.n_classes = n_classes
        self.n_layers = n_layers
        self.channel_independence = channel_independence
        self.downsampling_window = downsampling_window
        self.downsampling_layers = downsampling_layers
        self.downsampling_method = downsampling_method
        self.use_future_temporal_feature = use_future_temporal_feature

        self.pdm_blocks = nn.ModuleList(
            [
                PastDecomposableMixing(
                    n_steps,
                    0,  # n_pred_steps, 对于分类任务设为0
                    d_model,
                    d_ffn,
                    dropout,
                    channel_independence,
                    decomp_method,
                    top_k,
                    moving_avg,
                    downsampling_layers,
                    downsampling_window,
                )
                for _ in range(n_layers)
            ]
        )
        self.preprocess = SeriesDecompositionBlock(moving_avg)

        if self.channel_independence:
            self.enc_embedding = DataEmbedding(1, d_model, embed, freq, dropout, with_pos=False)
        else:
            self.enc_embedding = DataEmbedding(n_features, d_model, embed, freq, dropout, with_pos=False)

        self.normalize_layers = torch.nn.ModuleList([RevIN(n_features) for _ in range(downsampling_layers + 1)])

        if task_name == "classification":
            self.act = F.gelu
            self.dropout_layer = nn.Dropout(dropout)
            self.projection = nn.Linear(d_model * n_steps, n_classes)

    def classification(self, x_enc, x_mark_enc=None):
        x_enc, _ = self.__multi_scale_process_inputs(x_enc, None)
        x_list = x_enc

        # embedding
        enc_out_list = []
        for x in x_list:
            enc_out = self.enc_embedding(x, None)  # [B,T,C]
            enc_out_list.append(enc_out)

        # Past Decomposable Mixing
        for i in range(self.n_layers):
            enc_out_list = self.pdm_blocks[i](enc_out_list)

        enc_out = enc_out_list[0]
        # Output
        # the output transformer encoder/decoder embeddings don't include non-linearity
        output = self.act(enc_out)
        output = self.dropout_layer(output)
        # (batch_size, n_steps * d_model)
        output = output.reshape(output.shape[0], -1)
        output = self.projection(output)  # (batch_size, num_classes)
        return output

    def __multi_scale_process_inputs(self, x_enc, x_mark_enc):
        # 当前不进行下采样，直接返回输入
        return [x_enc], None

    def forward(self, x):
        return self.classification(x)
