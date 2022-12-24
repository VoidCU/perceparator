import torch
import torch.nn as nn
from MultiheadAttention import MultiHeadAttention
from torch.nn.modules.module import Module
from torch.autograd import Variable
import math
import numpy as np


#Encoding layer
class Encoder(nn.Module):

    def __init__(self, L, N):

        super(Encoder, self).__init__()

        self.L = L  # Convolution nucleus

        self.N = N  # Output channel size

        self.Conv1d = nn.Conv1d(in_channels=1,
                                out_channels=self.N,
                                kernel_size=self.L,
                                stride=self.L//2,
                                padding=0,
                                bias=False)

        self.ReLU = nn.ReLU()

    def forward(self, x):

        x = self.Conv1d(x)

        x = self.ReLU(x)

        return x

#Decoding layer
class Decoder(nn.Module):

    def __init__(self, L, N):

        super(Decoder, self).__init__()

        self.L = L

        self.N = N

        self.ConvTranspose1d = nn.ConvTranspose1d(in_channels=self.N,
                                                  out_channels=1,
                                                  kernel_size=self.L,
                                                  stride=self.L//2,
                                                  padding=0,
                                                  bias=False)

    def forward(self, x):

        x = self.ConvTranspose1d(x)

        return x

#Positional encoding
class Positional_Embedding(nn.Module):
    def __init__(self, input_shape, input_channels, embed_dim, bands = 4):
        super().__init__()

        self.ff = self.fourier_features(input_shape, bands = bands)
        self.conv = nn.Conv1d(input_channels + self.ff.shape[1], embed_dim, 1)

    def forward(self, x):
        enc = self.ff.unsqueeze(0).expand((x.shape[0],) + self.ff.shape)
        enc = enc.type_as(x)

        x = torch.cat([x, enc], dim =1)

        x = x.flatten(2)

        x = self.conv(x)

        x = x.permute(2, 0, 1)

        return x

    def fourier_features(self, shape, bands):
        
        dims = len(shape)

        pos = torch.stack(
            list(
                torch.meshgrid(
                    *(torch.linspace(-1.0, 1.0, steps = n) for n in list(shape))
                )
            )
        )
        pos = pos.unsqueeze(0).expand((bands,) + pos.shape)

        band_frequencies = (
            (
                torch.logspace(
                    math.log(1.0), math.log(shape[0] / 2), steps = bands, base = math.e
                )
            )
            .view((bands,) + tuple( 1 for _ in pos.shape[1:]))
            .expand(pos.shape)
        )

        result = (band_frequencies * math.pi * pos).view((dims * bands,) + shape)

        result = torch.cat(
            [
                torch.sin(result),
                torch.cos(result),
            ],
            dim = 0,
        )

        return result

class PerceiverAttention(nn.Module):
    def __init__(self, embed_dim, mlp_dim, n_heads, dropout = 0):
        super().__init__()

        self.lnorm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim= embed_dim, num_heads=n_heads)
        self.lnorm2 = nn.LayerNorm(embed_dim)
        self.linear1 = nn.Linear(embed_dim, mlp_dim)
        self.act = nn.GELU()
        self.linear2 = nn.Linear(mlp_dim, embed_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, q):
        
        q = self.lnorm1(q)

        out = self.attn(query =q, key = x, value =x)

        resid = out + q

        out = self.lnorm2(resid)
        out = self.linear1(out)
        out = self.act(out)
        out = self.linear2(out)
        out = self.drop(out)

        out = out + resid

        return out

class PerceiverBlock(nn.Module):
    def __init__(self, embed_dim, attn_mlp_dim, trnfr_mlp_dim, trnfr_heads, dropout, trnfr_layers):
        super().__init__()

        self.cross_attention = PerceiverAttention(embed_dim, attn_mlp_dim, trnfr_heads, dropout)
        self.latent_transformer = LatentTransformer(embed_dim, trnfr_mlp_dim, trnfr_heads, dropout, trnfr_layers)

    def forward(self, x, l):
        print(l.shape)
        print(x.shape)
        l = self.cross_attention(x, l)
        #print(l.shape)

        l = self.latent_transformer(l)
        
        return l

class LatentTransformer(nn.Module):
    def __init__(self, embed_dim, mlp_dim, n_heads, dropout, n_layers):
        super().__init__()

        self.transformer = nn.ModuleList(
            [
                PerceiverAttention(embed_dim=embed_dim, mlp_dim=mlp_dim,n_heads=n_heads,dropout=dropout)
                for l in range(n_layers)
            ]
        )

    def forward(self, l):
        for trnfr in self.transformer:
            l = trnfr(l, l)

        return l

class Perceiver(nn.Module):
    def __init__(self, input_shape, latent_dim, embed_dim, attn_mlp_dim, trnfr_mlp_dim, trnfr_heads, dropout, trnfr_layers, n_blocks):
        super().__init__()

        self.latent = nn.parameter.Parameter(
            torch.nn.init.trunc_normal_(
                torch.zeros(
                    (latent_dim, 1, embed_dim)), mean = 0, std = 0.02, a = -2, b = 2)
                    )

        self.embed = Positional_Embedding(input_shape, 1, embed_dim)

        self.perceiver_blocks = nn.ModuleList(
            [
                PerceiverBlock(
                    embed_dim=embed_dim,
                    attn_mlp_dim=attn_mlp_dim,
                    trnfr_mlp_dim=trnfr_mlp_dim,
                    trnfr_heads=trnfr_heads,
                    dropout=dropout,
                    trnfr_layers=trnfr_layers
                )
                for b in range(n_blocks)
            ]
        )

    def forward(self, x):
        batch_size = x.shape[1]
        latent = self.latent.expand(-1, batch_size, -1)
        x = self.embed(x)

        for pb in self.perceiver_blocks:
            latent = pb(x, latent).permute(1,0,2)

        return latent

class DPTBlock(nn.Module):

    def __init__(self, input_shape, latent_dim,embed_dim, attn_mlp_dim, trnfr_mlp_dim, trnfr_heads, dropout, trnfr_layers, n_blocks):

        super(DPTBlock, self).__init__()

        self.intra_transformer = Perceiver(input_shape=(250,66),latent_dim=latent_dim, embed_dim = embed_dim,attn_mlp_dim = attn_mlp_dim, trnfr_mlp_dim= trnfr_mlp_dim, trnfr_heads = trnfr_heads, dropout = dropout, trnfr_layers = trnfr_layers, n_blocks = n_blocks)

        self.inter_transformer = Perceiver(input_shape=(250,66),latent_dim=latent_dim, embed_dim = embed_dim,attn_mlp_dim = attn_mlp_dim, trnfr_mlp_dim= trnfr_mlp_dim, trnfr_heads = trnfr_heads, dropout = dropout, trnfr_layers = trnfr_layers, n_blocks = n_blocks)

    def forward(self, z):

        B, N, K, P = z.shape  # torch.Size([1, 64, 250, 130])

        # intra DPT
        row_z = z.permute(0, 3, 2, 1).reshape(B*P, K, N)
        row_z = self.intra_transformer(row_z.permute(1, 0, 2)).permute(1, 0, 2)
        row_output = row_z.reshape(B, P, K, N).permute(0, 3, 2, 1)

        # inter DPT
        col_z = row_output.permute(0, 2, 3, 1).reshape(B*K, P, N)
        col_z = self.inter_transformer(col_z.permute(1, 0, 2)).permute(1, 0, 2)
        col_output = col_z.reshape(B, K, P, N).permute(0, 3, 1, 2)

        return col_output

class Separator(nn.Module):

    def __init__(self, input_shape, latent_dim, embed_dim, attn_mlp_dim, trnfr_mlp_dim, C, trnfr_heads, trnfr_layers, K, Global_B, Local_B):

        super(Separator, self).__init__()

        self.input_shape = input_shape
        self.latent_dim = latent_dim
        self.embed_dim = embed_dim 
        self.attn_mlp_dim = attn_mlp_dim 
        self.trnfr_mlp_dim = trnfr_mlp_dim
        self.C = C
        self.trnfr_heads = trnfr_heads
        self.trnfr_layers = trnfr_layers
        self.K = K
        self.Global_B = Global_B  #Arcellar cycle
        self.Local_B = Local_B  #Plopenclastic number of cycles

        self.DPT = nn.ModuleList([])
        for i in range(self.Global_B):
            self.DPT.append(
                DPTBlock
                (
                    input_shape = self.input_shape, 
                    latent_dim = self.latent_dim,
                    embed_dim=self.embed_dim, 
                    attn_mlp_dim=self.attn_mlp_dim, 
                    trnfr_mlp_dim=self.trnfr_mlp_dim, 
                    trnfr_heads = self.trnfr_heads, 
                    dropout = 0, 
                    trnfr_layers = self.trnfr_layers, 
                    n_blocks = self.Local_B
                )
            )

        self.LayerNorm = nn.LayerNorm(self.input_shape)
        self.Linear1 = nn.Linear(in_features=self.input_shape, out_features=self.input_shape, bias=None)

        self.PReLU = nn.PReLU()
        self.Linear2 = nn.Linear(
            in_features=self.input_shape, out_features=self.input_shape*2, bias=None)

        self.FeedForward1 = nn.Sequential(nn.Linear(self.input_shape, self.input_shape*2*2),
                                          nn.ReLU(),
                                          nn.Linear(self.input_shape*2*2, self.input_shape))
        self.FeedForward2 = nn.Sequential(nn.Linear(self.input_shape, self.input_shape*2*2),
                                          nn.ReLU(),
                                          nn.Linear(self.input_shape*2*2, self.input_shape))
        self.ReLU = nn.ReLU()

    def forward(self, x):
        # Norm + Linear
        x = self.LayerNorm(x.permute(0, 2, 1))
        x = self.Linear1(x).permute(0, 2, 1)

        # Chunking
        # Block，[B, N, I] -> [B, N, K, S]
        out, gap = self.split_feature(x, self.K)

        # Perceiver
        for i in range(self.Global_B):
            self.input_shape = out.shape
            out = self.DPT[i](out)  # [B, N, K, S] -> [B, N, K, S]

        # PReLU + Linear
        out = self.PReLU(out)
        out = self.Linear2(out.permute(0, 3, 2, 1)).permute(0, 3, 2, 1)

        B, _, K, S = out.shape

        # OverlapAdd
        # [B, N*C, K, S] -> [B, N, C, K, S]
        out = out.reshape(B, -1, self.C, K, S).permute(0, 2, 1, 3, 4)
        out = out.reshape(B * self.C, -1, K, S)
        out = self.merge_feature(out, gap)  # [B*C, N, K, S]  -> [B*C, N, I]

        # FFW + ReLU
        out = self.FeedForward1(out.permute(0, 2, 1))
        out = self.FeedForward2(out).permute(0, 2, 1)
        out = self.ReLU(out)

        return out

    def pad_segment(self, input, segment_size):

        # Input feature: (B, N, T)

        batch_size, dim, seq_len = input.shape
        segment_stride = segment_size // 2

        rest = segment_size - (segment_stride + seq_len %
                               segment_size) % segment_size

        if rest > 0:
            pad = Variable(torch.zeros(batch_size, dim, rest)
                           ).type(input.type())
            input = torch.cat([input, pad], 2)

        pad_aux = Variable(torch.zeros(
            batch_size, dim, segment_stride)).type(input.type())

        input = torch.cat([pad_aux, input, pad_aux], 2)

        return input, rest

    def split_feature(self, input, segment_size):

        # Divide the features into pieces of section sizefeatures into pieces of section size
        # Input feature: (B, N, T)

        input, rest = self.pad_segment(input, segment_size)
        batch_size, dim, seq_len = input.shape
        segment_stride = segment_size // 2

        segments1 = input[:, :, :-segment_stride].contiguous().view(batch_size,
                                                                    dim, -1, segment_size)
        segments2 = input[:, :, segment_stride:].contiguous().view(
            batch_size, dim, -1, segment_size)
        segments = torch.cat([segments1, segments2], 3).view(
            batch_size, dim, -1, segment_size).transpose(2, 3)

        return segments.contiguous(), rest

    def merge_feature(self, input, rest):

        # Merge the characteristics of segments into a complete discourse
        # Input feature: (B, N, L, K)

        batch_size, dim, segment_size, _ = input.shape
        segment_stride = segment_size // 2
        input = input.transpose(2, 3).contiguous().view(
            batch_size, dim, -1, segment_size * 2)  # B, N, K, L

        input1 = input[:, :, :, :segment_size].contiguous().view(
            batch_size, dim, -1)[:, :, segment_stride:]
        input2 = input[:, :, :, segment_size:].contiguous().view(
            batch_size, dim, -1)[:, :, :-segment_stride]

        output = input1 + input2

        if rest > 0:
            output = output[:, :, :-rest]

        return output.contiguous()  # B, N, T

class Perceparator(nn.Module):
    """
        Args:
            C: Number of speakers
            N: Number of filters in autoencoder
            L: Length of the filters in autoencoder
            H: Multi-head
            K: segment size
            R: Number of repeats
    """

    def __init__(self, N=64, C=2, L=4, H=4, K=250, Global_B=2, Local_B=4):

        super(Perceparator, self).__init__()

        self.N = N  # Code output channel
        self.latent_dim = 8
        self.embed_dim = 64
        self.attn_mlp_dim = 16
        self.trnfr_mlp_dim = 16
        self.C = C  # The number of separation sources
        self.L = L  # Coder convolution core size
        self.trnfr_heads = H  # Pay attention to the number
        self.trnfr_layers = 6
        self.K = K  # Block size
        self.Global_B = Global_B  # Global cycle number
        self.Local_B = Local_B  # Local cycle number

        self.encoder = Encoder(self.L, self.N)

        self.separator = Separator(
            self.N,
            self.latent_dim,
            self.embed_dim,
            self.attn_mlp_dim,
            self.trnfr_mlp_dim,
            self.C, 
            self.trnfr_heads, 
            self.trnfr_layers,
            self.K, 
            self.Global_B, 
            self.Local_B)

        self.decoder = Decoder(self.L, self.N)

    def forward(self, x):

        # Encoding
        # Make up for zero，torch.Size([1, 1, 8000])
        print(type(x))
        x = x.unsqueeze(0)
        x, rest = self.pad_signal(x)
        # [B, 1, T] -> [B, N, I]，torch.Size([1, 64, 16002])

        enc_out = self.encoder(x)
        # Mask estimation
        # [B, N, I] -> [B*C, N, I]，torch.Size([2, 64, 16002])
        masks = self.separator(enc_out)

        _, N, I = masks.shape

        # [C, B, N, I]，torch.Size([2, 1, 64, 16002])
        masks = masks.view(self.C, -1, N, I)

        # Masking
        # C * ([B, N, I]) * [B, N, I]
        out = [masks[i] * enc_out for i in range(self.C)]

        # Decoding
        audio = [self.decoder(out[i]) for i in range(self.C)]  # C * [B, 1, T]

        audio[0] = audio[0][:, :, self.L // 2:-
                            (rest + self.L // 2)].contiguous()  # B, 1, T
        audio[1] = audio[1][:, :, self.L // 2:-
                            (rest + self.L // 2)].contiguous()  # B, 1, T
        audio = torch.cat(audio, dim=1)  # [B, C, T]

        return audio

    def pad_signal(self, input):

        # Enter waveform: (B, T) or (B, 1, T)
        # Adjust and fill

        if input.dim() not in [2, 3]:
            raise RuntimeError("Input can only be 2 or 3 dimensional.")

        if input.dim() == 2:
            input = input.unsqueeze(1)

        batch_size = input.size(0)  # The size of each batch
        nsample = input.size(2)  # The length of a single data
        rest = self.L - (self.L // 2 + nsample % self.L) % self.L

        if rest > 0:
            pad = Variable(torch.zeros(batch_size, 1, rest)).type(input.type())
            input = torch.cat([input, pad], dim=2)

        pad_aux = Variable(torch.zeros(
            batch_size, 1, self.L // 2)).type(input.type())

        input = torch.cat([pad_aux, input, pad_aux], 2)

        return input, rest

    @classmethod
    def load_model(cls, path):

        package = torch.load(path, map_location=lambda storage, loc: storage)

        model = cls.load_model_from_package(package)

        return model

    @classmethod
    def load_model_from_package(cls, package):

        model = cls(N=package['N'], C=package['C'], L=package['L'],
                    H=package['H'], K=package['K'], Global_B=package['Global_B'],
                    Local_B=package['Local_B'])

        model.load_state_dict(package['state_dict'])

        return model

    @staticmethod
    def serialize(model, optimizer, epoch, tr_loss=None, cv_loss=None):

        package = {
            # hyper-parameter
            'N': model.N, 'C': model.C, 'L': model.L,
            'H': model.H, 'K': model.K, 'Global_B': model.Global_B,
            'Local_B': model.Local_B,

            # state
            'state_dict': model.state_dict(),
            'optim_dict': optimizer.state_dict(),
            'epoch': epoch
        }

        if tr_loss is not None:
            package['tr_loss'] = tr_loss
            package['cv_loss'] = cv_loss

        return package

if __name__ == "__main__":

    x = torch.rand(1, 8000)

    model = Perceparator(N= 256,
                        C=2,
                        L=2,
                        H=8,
                        K=250,
                        Global_B=2,
                        Local_B=8)

    print("{:.3f} million".format(
        sum([param.nelement() for param in model.parameters()]) / 1e6))

    y = model(x)

    print(y.shape)
