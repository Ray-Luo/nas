import torch
import torch.nn as nn

import torch
import torch.nn as nn
import torch.nn.functional as F

BASE_LATENCY = 10

class InvertedResidualBase(nn.Module):
    def __init__(
        self,
        inp,
        oup,
        stride,
        expansion=6,
        out_channel_mask=None,
    ):
        super(InvertedResidualBase, self).__init__()
        assert stride in [1, 2]

        hidden_dim = expansion * inp

        self.identity = stride == 1 and inp == oup

        self.out_channel_mask = out_channel_mask
        self.expansion_mask = nn.Parameter(torch.ones(hidden_dim, requires_grad=True), requires_grad=True)

        self.act_mask = nn.Parameter(torch.zeros(1, requires_grad=True))

        self.relu = nn.ReLU(inplace=False)

        self.se_mask = nn.Parameter(torch.zeros(1, requires_grad=True))
        self.id = nn.Identity()

        self.pw = nn.Sequential(
            nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
            nn.BatchNorm2d(hidden_dim),
        )

        self.pw_linear = nn.Sequential(
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        )

    def forward(self, x):
        latency = 0

        expansion_mask = torch.round(torch.sigmoid(self.expansion_mask)).view(
            1, -1, 1, 1
        )
        expansion_sum = torch.sum(expansion_mask)

        # point_wise conv

        out = self.pw(x)
        latency += x.shape[1] * expansion_sum * BASE_LATENCY # pw
        latency += expansion_sum * BASE_LATENCY # bn
        latency += BASE_LATENCY # act

        return out, latency


target = torch.randn(1,3,512,512).cuda()
input = torch.randn(1, 3, 512, 512).cuda()
model = InvertedResidualBase(3,3,1).cuda()
out, latency_original = model(input)

import torch.optim as optim

optimizer = optim.SGD(model.parameters(), lr=1e-3)
criterion = nn.L1Loss()
num_epochs = 1000
weight = 1e-8

for epoch in range(num_epochs):
    optimizer.zero_grad()

    out, latency = model(input)

    loss = latency * weight

    print("Epoch: {}, Loss: {}, Lat: {}, Ori_lat: {}".format(epoch, loss.item(), latency.item(), latency_original.item()))
    loss.backward()
    print("******", model.expansion_mask.grad)
    optimizer.step()
