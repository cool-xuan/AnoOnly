import torch
from torch import nn

class prenet(nn.Module):
    def __init__(self, input_size, act_fun):
        super(prenet, self).__init__()

        self.feature = nn.Sequential(
            nn.Linear(input_size, 20),
            act_fun
        )

        self.reg = nn.Linear(40, 1)

    #the input vector of prenet should be a pair
    def forward(self, X_left, X_right):
        feature_left = self.feature(X_left)
        feature_right = self.feature(X_right)

        # concat feature
        feature = torch.cat((feature_left, feature_right), dim=1)
        # generate score based on the concat feature
        score = self.reg(feature)

        return score.squeeze()
    
class prenet_anoonly(nn.Module):
    def __init__(self, input_size, act_fun):
        super(prenet_anoonly, self).__init__()

        self.feature = nn.Sequential(
            # nn.BatchNorm1d(input_size),
            nn.Linear(input_size, 20),
            nn.BatchNorm1d(20),
            act_fun,
            nn.BatchNorm1d(20),
        )

        self.reg = nn.Linear(40, 1)

    #the input vector of prenet should be a pair
    def forward(self, X_left, X_right, y=None):
        feature_left = self.feature(X_left)
        feature_right = self.feature(X_right)
        # if y is not None:
        #     feature_right[y==4] = feature_right[y==4].detach()
        # concat feature
        feature = torch.cat((feature_left, feature_right), dim=1)
        # generate score based on the concat feature
        score = self.reg(feature)

        return score.squeeze()