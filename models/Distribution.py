from torch import nn
import torch

class Gaussian(nn.Module):

    def __init__(self, hidden_size, output_size):
        '''
        Gaussian Likelihood Supports Continuous Data
        Args:
        input_size (int): hidden h_{i,t} column size
        output_size (int): embedding size
        '''
        super(Gaussian, self).__init__()
        self.mu_layer = nn.Linear(hidden_size, output_size)
        self.sigma_layer = nn.Linear(hidden_size, output_size)

        # initialize weights
        # nn.init.xavier_uniform_(self.mu_layer.weight)
        # nn.init.xavier_uniform_(self.sigma_layer.weight)

    def forward(self, dec_out):
        sigma_t = torch.log(1 + torch.exp(self.sigma_layer(dec_out))) + 1e-6
        mu_t = self.mu_layer(dec_out)
        return mu_t, sigma_t


class NegativeBinomial(nn.Module):

    def __init__(self, input_size, output_size):
        '''
        Negative Binomial Supports Positive Count Data
        Args:
        input_size (int): hidden h_{i,t} column size
        output_size (int): embedding size
        '''
        super(NegativeBinomial, self).__init__()
        self.mu_layer = nn.Linear(input_size, output_size)
        self.sigma_layer = nn.Linear(input_size, output_size)

    def forward(self, dec_out):
        alpha_t = torch.log(1 + torch.exp(self.sigma_layer(dec_out))) + 1e-6
        mu_t = torch.log(1 + torch.exp(self.mu_layer(dec_out)))
        return mu_t, alpha_t