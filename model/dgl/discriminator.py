import torch
import torch.nn as nn
import time 
class Discriminator(nn.Module):
    r""" Discriminator module for calculating MI"""
    
    def __init__(self, n_e, n_g):
        """
        param: n_e: dimension of edge embedding
        param: n_g: dimension of graph embedding
        """
        super(Discriminator, self).__init__()
        self.f_k = nn.Bilinear(n_e, n_g, 1)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Bilinear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, c, h_pl, h_mi, s_bias1=None, s_bias2=None):
        c_x = torch.unsqueeze(c, 0) # [1, F]
        c_x = c_x.expand_as(h_pl)   #[B, F]

        sc_1 = torch.squeeze(self.f_k(h_pl, c_x), 1) # [B];  self.f_k(h_pl, c_x): [B, 1]
        sc_2 = torch.squeeze(self.f_k(h_mi, c_x), 1) # [B]

        # print('Discriminator time:', time.time() - ts)
        if s_bias1 is not None:
            sc_1 += s_bias1
        if s_bias2 is not None:
            sc_2 += s_bias2

        logits = torch.cat((sc_1, sc_2))

        return logits
