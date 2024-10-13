import torch
import torch.nn as nn

from .autoformer import SeriesDecompositionBlock


# 如果需要，请将 SeriesDecompositionBlock 类包含在此文件中
# from ..autoformer import SeriesDecompositionBlock


class DFT_series_decomp(nn.Module):
    """
    Series decomposition block
    """

    def __init__(self, top_k=5):
        super().__init__()
        self.top_k = top_k

    def forward(self, x):
        xf = torch.fft.rfft(x)
        freq = abs(xf)
        freq[..., 0] = 0
        top_k_freq, top_list = torch.topk(freq, self.top_k, dim=-1)
        xf[freq <= top_k_freq.min(dim=-1, keepdim=True).values] = 0
        x_season = torch.fft.irfft(xf, n=x.size(-1))
        x_trend = x - x_season
        return x_season, x_trend


class MultiScaleSeasonMixing(nn.Module):
    """
    Bottom-up mixing season pattern
    """

    def __init__(
        self,
        n_steps,
        downsampling_window,
        downsampling_layers,
    ):
        super().__init__()

        self.downsampling_layers = torch.nn.ModuleList(
            [
                nn.Sequential(
                    torch.nn.Linear(
                        n_steps // (downsampling_window ** i),
                        n_steps // (downsampling_window ** (i + 1)),
                    ),
                    nn.GELU(),
                    torch.nn.Linear(
                        n_steps // (downsampling_window ** (i + 1)),
                        n_steps // (downsampling_window ** (i + 1)),
                    ),
                )
                for i in range(downsampling_layers)
            ]
        )

    def forward(self, season_list):

        # mixing high->low
        out_high = season_list[0]
        out_season_list = [out_high.permute(0, 2, 1)]

        for i in range(len(season_list) - 1):
            out_low_res = self.downsampling_layers[i](out_high)
            out_high = season_list[i + 1] + out_low_res
            out_season_list.append(out_high.permute(0, 2, 1))

        return out_season_list


class MultiScaleTrendMixing(nn.Module):
    """
    Top-down mixing trend pattern
    """

    def __init__(
        self,
        n_steps,
        downsampling_window,
        downsampling_layers,
    ):
        super().__init__()

        self.up_sampling_layers = torch.nn.ModuleList(
            [
                nn.Sequential(
                    torch.nn.Linear(
                        n_steps // (downsampling_window ** (i + 1)),
                        n_steps // (downsampling_window ** i),
                    ),
                    nn.GELU(),
                    torch.nn.Linear(
                        n_steps // (downsampling_window ** i),
                        n_steps // (downsampling_window ** i),
                    ),
                )
                for i in reversed(range(downsampling_layers))
            ]
        )

    def forward(self, trend_list):

        # mixing low->high
        trend_list_reverse = trend_list[::-1]
        out_low = trend_list_reverse[0]
        out_trend_list = [out_low.permute(0, 2, 1)]

        for i in range(len(trend_list_reverse) - 1):
            out_high_res = self.up_sampling_layers[i](out_low)
            out_low = trend_list_reverse[i + 1] + out_high_res
            out_trend_list.append(out_low.permute(0, 2, 1))

        out_trend_list.reverse()
        return out_trend_list


class PastDecomposableMixing(nn.Module):
    def __init__(
        self,
        n_steps,
        n_pred_steps,
        d_model,
        d_ffn,
        dropout,
        channel_independence,
        decomp_method,
        top_k,
        moving_avg,
        downsampling_layers,
        downsampling_window,
    ):
        super().__init__()
        self.n_steps = n_steps
        self.n_pred_steps = n_pred_steps
        self.downsampling_window = downsampling_window

        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.channel_independence = channel_independence

        if decomp_method == "moving_avg":
            self.decomposition = SeriesDecompositionBlock(moving_avg)
        elif decomp_method == "dft_decomp":
            self.decomposition = DFT_series_decomp(top_k)
        else:
            raise ValueError("decomposition method is invalid")

        if not channel_independence:
            self.cross_layer = nn.Sequential(
                nn.Linear(in_features=d_model, out_features=d_ffn),
                nn.GELU(),
                nn.Linear(in_features=d_ffn, out_features=d_model),
            )

        # Mixing season
        self.mixing_multi_scale_season = MultiScaleSeasonMixing(
            n_steps,
            downsampling_window,
            downsampling_layers,
        )

        # Mixing trend
        self.mixing_multi_scale_trend = MultiScaleTrendMixing(
            n_steps,
            downsampling_window,
            downsampling_layers,
        )

        self.out_cross_layer = nn.Sequential(
            nn.Linear(in_features=d_model, out_features=d_ffn),
            nn.GELU(),
            nn.Linear(in_features=d_ffn, out_features=d_model),
        )

    def forward(self, x_list):
        length_list = []
        for x in x_list:
            _, T, _ = x.size()
            length_list.append(T)

        # Decompose to obtain the season and trend
        season_list = []
        trend_list = []
        for x in x_list:
            season, trend = self.decomposition(x)
            if not self.channel_independence:
                season = self.cross_layer(season)
                trend = self.cross_layer(trend)
            season_list.append(season.permute(0, 2, 1))
            trend_list.append(trend.permute(0, 2, 1))

        # Bottom-up season mixing
        out_season_list = self.mixing_multi_scale_season(season_list)
        # Top-down trend mixing
        out_trend_list = self.mixing_multi_scale_trend(trend_list)

        out_list = []
        for ori, out_season, out_trend, length in zip(x_list, out_season_list, out_trend_list, length_list):
            out = out_season + out_trend
            if self.channel_independence:
                out = ori + self.out_cross_layer(out)
            out_list.append(out[:, :length, :])

        return out_list
