import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding_inverted
import numpy as np

class Expert(nn.Module):
    def __init__(self, d_model, num_experts, device):
        super(Expert, self).__init__()
        self.num_experts = num_experts
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.ReLU(),
                nn.Linear(d_model, d_model)
            ).to(device) for _ in range(num_experts)
        ])

    def forward(self, x):
        # x shape: [batch_size, seq_len, d_model]
        batch_size, seq_len, d_model = x.shape
        # Apply each expert to the input
        expert_outputs = []
        for expert in self.experts:
            expert_outputs.append(expert(x))
        # Stack expert outputs along a new dimension
        expert_outputs = torch.stack(expert_outputs, dim=0)  # [num_experts, batch_size, seq_len, d_model]
        return expert_outputs
    

class GatingNetwork(nn.Module):
    def __init__(self, d_model, num_experts, device):
        super(GatingNetwork, self).__init__()
        self.gating = nn.Sequential(
            nn.Linear(d_model, num_experts),
            nn.Softmax(dim=-1)
        ).to(device)

    def forward(self, x):
        # x shape: [batch_size, seq_len, d_model]
        gating_weights = self.gating(x)  # [batch_size, seq_len, num_experts]
        return gating_weights


class MoE(nn.Module):
    def __init__(self, d_model, num_experts, device):
        super(MoE, self).__init__()
        self.expert = Expert(d_model, num_experts, device)
        self.gating = GatingNetwork(d_model, num_experts, device)

    def forward(self, x):
        # x shape: [batch_size, seq_len, d_model]
        expert_outputs = self.expert(x)  # [num_experts, batch_size, seq_len, d_model]
        gating_weights = self.gating(x)  # [batch_size, seq_len, num_experts]

        # Apply gating weights to expert outputs
        moe_output = torch.einsum('bsn, nbsd ->bsd', gating_weights, expert_outputs)
        return moe_output
    

class Model(nn.Module):
    """
    Paper link: https://arxiv.org/abs/2310.06625
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention
        # Embedding
        self.enc_embedding = DataEmbedding_inverted(configs.seq_len, configs.d_model, configs.embed, configs.freq,
                                                    configs.dropout)
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=configs.output_attention), configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )
        # Decoder
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            self.projection = nn.Linear(configs.d_model, configs.pred_len, bias=True)
        if self.task_name == 'imputation':
            self.projection = nn.Linear(configs.d_model, configs.seq_len, bias=True)
        if self.task_name == 'anomaly_detection':
            self.projection = nn.Linear(configs.d_model, configs.seq_len, bias=True)
        if self.task_name == 'classification':
            self.act = F.gelu
            self.dropout = nn.Dropout(configs.dropout)
            # self.projection = nn.Linear(configs.d_model * configs.enc_in, configs.num_class)
            self.projection = nn.Linear(49152, configs.num_class)

            self.multiscale_conv = nn.ModuleList([
                nn.Sequential(
                    nn.Conv1d(configs.enc_in, configs.d_model, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm1d(configs.d_model),
                    nn.LeakyReLU(inplace=True)
                ),
                nn.Sequential(
                    nn.Conv1d(configs.enc_in, configs.d_model, kernel_size=5, stride=1, padding=2),
                    nn.BatchNorm1d(configs.d_model),
                    nn.LeakyReLU(inplace=True)
                ),
                nn.Sequential(
                    nn.Conv1d(configs.enc_in, configs.d_model, kernel_size=7, stride=1, padding=3),
                    nn.BatchNorm1d(configs.d_model),
                    nn.LeakyReLU(inplace=True)
                )
            ])

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        _, _, N = x_enc.shape

        # Embedding
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        dec_out = self.projection(enc_out).permute(0, 2, 1)[:, :, :N]
        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        return dec_out

    def imputation(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask):
        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        _, L, N = x_enc.shape

        # Embedding
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        dec_out = self.projection(enc_out).permute(0, 2, 1)[:, :, :N]
        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, L, 1))
        dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, L, 1))
        return dec_out

    def anomaly_detection(self, x_enc):
        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        _, L, N = x_enc.shape

        # Embedding
        enc_out = self.enc_embedding(x_enc, None)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        dec_out = self.projection(enc_out).permute(0, 2, 1)[:, :, :N]
        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, L, 1))
        dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, L, 1))
        return dec_out

    # def classification(self, x_enc, x_mark_enc):
    #     # Embedding
    #     enc_out = self.enc_embedding(x_enc, None)
    #     enc_out, attns = self.encoder(enc_out, attn_mask=None)

    #     # Output
    #     output = self.act(enc_out)  # the output transformer encoder/decoder embeddings don't include non-linearity
    #     output = self.dropout(output)
    #     output = output.reshape(output.shape[0], -1)  # (batch_size, c_in * d_model)
    #     output = self.projection(output)  # (batch_size, num_classes)
    #     return output

    def classification(self, x_enc, x_mark_enc):
        # Multi-scale convolution
        x_enc = x_enc.permute(0, 2, 1)  # [batch_size, enc_in, seq_len]
        conv_outs = []
        for conv in self.multiscale_conv:
            conv_out = conv(x_enc)
            conv_outs.append(conv_out)
        x_enc = torch.cat(conv_outs, dim=1)  # [batch_size, d_model * num_scales, seq_len]
        x_enc = x_enc.permute(0, 2, 1)  # [batch_size, seq_len, d_model * num_scales]

        # Embedding
        enc_out = self.enc_embedding(x_enc, None)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        # Ensure the device is consistent
        device = enc_out.device
        # MoE module
        moe = MoE(128, num_experts=3, device=device)  # 假设使用4个专家
        moe_output = moe(enc_out)

        # Output
        output = self.act(moe_output)  # the output transformer encoder/decoder embeddings don't include non-linearity
        output = self.dropout(output)
        output = output.reshape(output.shape[0], -1)  # (batch_size, c_in * d_model)
        
        if output.shape[1] != self.projection.in_features:
            raise ValueError(f"Input dimension mismatch: expected {self.projection.in_features}, got {output.shape[1]}")
        
        output = self.projection(output)  # (batch_size, num_classes)
        return output

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]
        if self.task_name == 'imputation':
            dec_out = self.imputation(x_enc, x_mark_enc, x_dec, x_mark_dec, mask)
            return dec_out  # [B, L, D]
        if self.task_name == 'anomaly_detection':
            dec_out = self.anomaly_detection(x_enc)
            return dec_out  # [B, L, D]
        if self.task_name == 'classification':
            dec_out = self.classification(x_enc, x_mark_enc)
            return dec_out  # [B, N]
        return None
