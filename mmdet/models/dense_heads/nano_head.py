# Copyright (c) OpenMMLab. All rights reserved.

import torch
import torch.nn as nn

from mmcv.cnn import ConvModule, Scale, DepthwiseSeparableConvModule

from .gfl_head import GFLHead


class NanoDetHead(GFLHead):
    """NanoDetHead

    Modified from GFL, use same loss functions but much lightweight convolution heads.

    Sepecifically,
    - replace convolutional layer by depth-wise convolutional layer
    - Use BN instead of GN
    - Decrease featrue channels from 256 to 96
    - Decrease stacked_convs from 4 to 3

    """

    def __init__(self,
                 share_cls_reg=False,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 act_cfg=dict(type='LeakyReLU', negative_slope=0.1),
                 **kwargs):
        self.share_cls_reg = share_cls_reg
        self.act_cfg = act_cfg
        super(NanoDetHead, self).__init__(norm_cfg=norm_cfg,
                                          **kwargs)


    def _init_layers(self):
        """Initialize layers of the head."""

        self.scales = nn.ModuleList(
            [Scale(1.0) for _ in self.prior_generator.strides])

        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        for _ in self.scales:
            cls_convs, reg_convs = self._buid_not_shared_head()
            self.cls_convs.append(cls_convs)
            self.reg_convs.append(reg_convs)

        self.gfl_cls = nn.ModuleList(
            [
                nn.Conv2d(self.feat_channels,
                          self.cls_out_channels + 4 * (self.reg_max + 1)
                          if self.share_cls_reg
                          else self.cls_out_channels,
                          1,
                          padding=0
                )
                for _ in self.scales
            ]
        )

        self.gfl_reg = nn.ModuleList(
            [
                nn.Conv2d(self.feat_channels, 4 * (self.reg_max + 1), 1, padding=0)
                for _ in self.scales
            ]
        )

    def _buid_not_shared_head(self):
        cls_convs = nn.ModuleList()
        reg_convs = nn.ModuleList()
        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            cls_convs.append(
                DepthwiseSeparableConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg,
                    dw_act_cfg=None
                )
            )
            if not self.share_cls_reg:
                reg_convs.append(
                DepthwiseSeparableConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg,
                    dw_act_cfg=None
                )
            )
        return cls_convs, reg_convs


    def forward(self, feats):
        """Forward features from the upstream network.

        Args:
            feats (tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.

        Returns:
            tuple: Usually a tuple of classification scores and bbox prediction
                cls_scores (list[Tensor]): Classification and quality (IoU)
                    joint scores for all scale levels, each is a 4D-tensor,
                    the channel number is num_classes.
                bbox_preds (list[Tensor]): Box distribution logits for all
                    scale levels, each is a 4D-tensor, the channel number is
                    4*(n+1), n is max value of integral set.
        """
        cls_score_list = []
        bbox_pred_list = []
        for x, cls_convs, reg_convs, gfl_cls, gfl_reg in zip(
            feats, self.cls_convs, self.reg_convs, self.gfl_cls, self.gfl_reg
        ):
            cls_feat = x
            reg_feat = x
            for cls_conv in cls_convs:
                cls_feat = cls_conv(cls_feat)
            for reg_conv in reg_convs:
                reg_feat = reg_conv(reg_feat)

            if self.share_cls_reg:
                output = gfl_cls(cls_feat)
                cls_score, bbox_pred = torch.split(output,
                                                   [self.cls_out_channels, 4 * (self.reg_max + 1)],
                                                   dim=1)
            else:
                cls_score = gfl_cls(cls_feat)
                bbox_pred = gfl_reg(reg_feat)

            cls_score_list.append(cls_score)
            bbox_pred_list.append(bbox_pred)

        return cls_score_list, bbox_pred_list