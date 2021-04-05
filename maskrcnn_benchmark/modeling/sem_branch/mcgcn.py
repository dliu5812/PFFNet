# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
import torch.nn.functional as F
from torch import nn

from maskrcnn_benchmark.modeling.make_layers import conv_with_kaiming_uniform
from maskrcnn_benchmark.modeling.make_layers import conv4gcn_with_kaiming_uniform

def initialize_weights(*models):
    for model in models:
        for module in model.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear) or isinstance(module, nn.ConvTranspose2d):
                nn.init.kaiming_uniform_(module.weight, a=1)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm2d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()



class GCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GCNBlock, self).__init__()

        self.gcn_l1 = conv4gcn_with_kaiming_uniform()(in_channels, out_channels, kernel_size=(5, 1))
        self.gcn_l2 = conv4gcn_with_kaiming_uniform()(out_channels, out_channels, kernel_size=(1, 5))
        self.gcn_r1 = conv4gcn_with_kaiming_uniform()(in_channels, out_channels, kernel_size=(1, 5))
        self.gcn_r2 = conv4gcn_with_kaiming_uniform()(out_channels, out_channels, kernel_size=(5, 1))

    def forward(self, x):

        out_l1 = self.gcn_l1(x)
        out_r1 = self.gcn_r1(x)

        out_l = self.gcn_l2(out_l1)
        out_r = self.gcn_r2(out_r1)

        out = out_l + out_r

        return out


class RESBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        conv_block
    ):
        super(RESBlock, self).__init__()

        self.conv_1 = conv_block(in_channels, out_channels,3)

        self.conv_2 = conv_block(out_channels, out_channels,3)

    def forward(self, x):
        residual = x

        out = self.conv_1(x)
        out = self.conv_2(out)

        out += residual

        return out


class RGCN(nn.Module):
    """
    Module that adds FPN on top of a list of feature maps.
    The feature maps are currently supposed to be in increasing depth
    order, and must be consecutive
    """

    def __init__(self, in_channels_list, middle_channels, out_channels, conv_block):
        """
        Arguments:
            in_channels_list (list[int]): number of channels for each feature map that
                will be fed
            out_channels (int): number of channels of the FPN representation
            top_blocks (nn.Module or None): if provided, an extra operation will
                be performed on the output of the last (smallest resolution)
                FPN output, and the result will extend the result list
        """
        super(RGCN, self).__init__()
        self.inner_blocks = []
        self.layer_blocks = []
        self.gcn_blocks = []
        self.res_blocks = []
        self.res2_blocks = []

        self.sem_final_block = conv_with_kaiming_uniform()(middle_channels, out_channels, 1)

        for idx, in_channels in enumerate(in_channels_list, 1):
            inner_block = "fpn_inner{}".format(idx)
            layer_block = "fpn_layer{}".format(idx)
            gcn_block = "gcn_layer{}".format(idx)
            res_block = "res_layer{}".format(idx)
            res2_block = "res2_layer{}".format(idx)

            gcn_block_module = GCNBlock(in_channels, middle_channels)
            res_block_module = RESBlock(middle_channels, middle_channels, conv_block)
            res2_block_module = RESBlock(middle_channels, middle_channels, conv_block)

            inner_block_module = nn.Conv2d(in_channels, out_channels, 1)
            layer_block_module = nn.Conv2d(out_channels, out_channels, 3, 1, 1)


            self.add_module(inner_block, inner_block_module)
            self.add_module(layer_block, layer_block_module)
            self.add_module(gcn_block, gcn_block_module)
            self.add_module(res_block, res_block_module)
            self.add_module(res2_block, res2_block_module)

            self.inner_blocks.append(inner_block)
            self.layer_blocks.append(layer_block)
            self.gcn_blocks.append(gcn_block)
            self.res_blocks.append(res_block)
            self.res2_blocks.append(res2_block)



    def forward(self, x):
        """
        Arguments:
            x (list[Tensor]): feature maps for each feature level.
        Returns:
            results (tuple[Tensor]): feature maps after FPN layers.
                They are ordered from highest resolution first.
        """

        last_inner1 = getattr(self, self.gcn_blocks[-1])(x[-1])
        last_inner = getattr(self, self.res2_blocks[-1])(last_inner1)


        results = []
        results.append(last_inner)
        idx = 1
        for feature, inner_block, layer_block in zip(
            x[:-1][::-1], self.inner_blocks[:-1][::-1], self.layer_blocks[:-1][::-1]
        ):
            inner_top_down = F.interpolate(last_inner, scale_factor=2, mode="nearest")
            idx = idx + 1

            inner_lateral = getattr(self, self.gcn_blocks[-idx])(feature)
            inner_lateral2 = getattr(self, self.res2_blocks[-idx])(inner_lateral)

            last_inner_pre = inner_lateral2 + inner_top_down
            last_inner = getattr(self, self.res_blocks[-idx+1])(last_inner_pre)
            results.insert(0, last_inner)


        out_25 = results[0]
        sem_map_25 = self.sem_final_block(out_25)
        sem_map = F.interpolate(sem_map_25, scale_factor=4, mode="nearest")
        results.insert(0, sem_map)

        return sem_map






