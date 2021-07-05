import numpy as np
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from maskrcnn_benchmark.modeling.poolers import Pooler
from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.modeling.backbone import resnet


###
class Encoder(nn.Module):
    """
    DCGAN ENCODER NETWORK
    """

    def __init__(self, isize, nz, nc, ndf, ngpu, n_extra_layers=0, add_final_conv=True):
        super(Encoder, self).__init__()
        self.ngpu = ngpu
        assert isize % 16 == 0, "isize has to be a multiple of 16"

        main = nn.Sequential()
        # input is nc x isize x isize
        main.add_module('initial-conv-{0}-{1}'.format(nc, ndf),
                        nn.Conv2d(nc, ndf, 4, 2, 1, bias=False))
        main.add_module('initial-relu-{0}'.format(ndf),
                        nn.LeakyReLU(0.2, inplace=True))
        csize, cndf = isize / 2, ndf

        # Extra layers
        for t in range(n_extra_layers):
            main.add_module('extra-layers-{0}-{1}-conv'.format(t, cndf),
                            nn.Conv2d(cndf, cndf, 3, 1, 1, bias=False))
            main.add_module('extra-layers-{0}-{1}-batchnorm'.format(t, cndf),
                            nn.BatchNorm2d(cndf))
            main.add_module('extra-layers-{0}-{1}-relu'.format(t, cndf),
                            nn.LeakyReLU(0.2, inplace=True))

        while csize > 4:
            in_feat = cndf
            out_feat = cndf * 2
            main.add_module('pyramid-{0}-{1}-conv'.format(in_feat, out_feat),
                            nn.Conv2d(in_feat, out_feat, 4, 2, 1, bias=False))
            main.add_module('pyramid-{0}-batchnorm'.format(out_feat),
                            nn.BatchNorm2d(out_feat))
            main.add_module('pyramid-{0}-relu'.format(out_feat),
                            nn.LeakyReLU(0.2, inplace=True))
            cndf = cndf * 2
            csize = csize / 2

        # state size. K x 4 x 4
        if add_final_conv:
            main.add_module('final-{0}-{1}-conv'.format(cndf, 1),
                            nn.Conv2d(cndf, nz, 4, 1, 0, bias=False))

        self.main = main

    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)

        return output

##
class Decoder(nn.Module):
    """
    DCGAN DECODER NETWORK
    """
    def __init__(self, isize, nz, nc, ngf, ngpu, n_extra_layers=0):
        super(Decoder, self).__init__()
        self.ngpu = ngpu
        assert isize % 16 == 0, "isize has to be a multiple of 16"

        cngf, tisize = ngf // 2, 4
        while tisize != isize:
            cngf = cngf * 2
            tisize = tisize * 2

        main = nn.Sequential()
        # input is Z, going into a convolution
        main.add_module('initial-{0}-{1}-convt'.format(nz, cngf),
                        nn.ConvTranspose2d(nz, cngf, 4, 1, 0, bias=False))
        main.add_module('initial-{0}-batchnorm'.format(cngf),
                        nn.BatchNorm2d(cngf))
        main.add_module('initial-{0}-relu'.format(cngf),
                        nn.ReLU(True))

        csize, _ = 4, cngf
        while csize < isize // 2:
            main.add_module('pyramid-{0}-{1}-convt'.format(cngf, cngf // 2),
                            nn.ConvTranspose2d(cngf, cngf // 2, 4, 2, 1, bias=False))
            main.add_module('pyramid-{0}-batchnorm'.format(cngf // 2),
                            nn.BatchNorm2d(cngf // 2))
            main.add_module('pyramid-{0}-relu'.format(cngf // 2),
                            nn.ReLU(True))
            cngf = cngf // 2
            csize = csize * 2

        # Extra layers
        for t in range(n_extra_layers):
            main.add_module('extra-layers-{0}-{1}-conv'.format(t, cngf),
                            nn.Conv2d(cngf, cngf, 3, 1, 1, bias=False))
            main.add_module('extra-layers-{0}-{1}-batchnorm'.format(t, cngf),
                            nn.BatchNorm2d(cngf))
            main.add_module('extra-layers-{0}-{1}-relu'.format(t, cngf),
                            nn.ReLU(True))

        main.add_module('final-{0}-{1}-convt'.format(cngf, nc),
                        nn.ConvTranspose2d(cngf, nc, 4, 2, 1, bias=False))
        main.add_module('final-{0}-tanh'.format(nc),
                        nn.Tanh())
        self.main = main

    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        return output

class PoolHead(torch.nn.Module):
    def __init__(self, fastercnn_config, image_height=256, image_width=256):
        super(PoolHead, self).__init__()
        self.image_height = image_height
        self.image_width = image_width

        resolution = fastercnn_config.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        scales = fastercnn_config.MODEL.ROI_BOX_HEAD.POOLER_SCALES
        sampling_ratio = fastercnn_config.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        pooler = Pooler(
            output_size=(resolution, resolution),
            scales=scales,
            sampling_ratio=sampling_ratio,
        )

        stage = resnet.StageSpec(index=4, block_count=3, return_features=False)
        head = resnet.ResNetHead(
            block_module=fastercnn_config.MODEL.RESNETS.TRANS_FUNC,
            stages=(stage,),
            num_groups=fastercnn_config.MODEL.RESNETS.NUM_GROUPS,
            width_per_group=fastercnn_config.MODEL.RESNETS.WIDTH_PER_GROUP,
            stride_in_1x1=fastercnn_config.MODEL.RESNETS.STRIDE_IN_1X1,
            stride_init=None,
            res2_out_channels=fastercnn_config.MODEL.RESNETS.RES2_OUT_CHANNELS,
            dilation=fastercnn_config.MODEL.RESNETS.RES5_DILATION
        )

        self.pooler = pooler
        self.head = head
        self.avgpool = nn.AdaptiveAvgPool2d(1)

    def forward(self, feats, bboxes):
        bboxes = BoxList(bboxes, (self.image_height, self.image_width), mode="xyxy")
        x = self.pooler([feats, ], [bboxes, ])
        x = self.head(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return x


class TemporalTransformer(torch.nn.Module):
    def __init__(self):
        super(TemporalTransformer, self).__init__()

    def forward(self, object_feat, feats):

        pass

class VadTransformer(torch.nn.Module):
    def __init__(self, fastercnn_config, image_height=256, image_width=256, n_channel =3,  t_length = 5):
        super(VadTransformer, self).__init__()
        self.config = fastercnn_config
        self.image_height = image_height
        self.image_width = image_width

        resolution = 14
        scales = (1.0 / 16,)
        sampling_ratio = 0
        self.pooler = Pooler(
            output_size=(resolution, resolution),
            scales=scales,
            sampling_ratio=sampling_ratio,
        )

        stage = resnet.StageSpec(index=4, block_count=3, return_features=False)
        self.head = resnet.ResNetHead(
            block_module=self.config.MODEL.RESNETS.TRANS_FUNC,
            stages=(stage,),
            num_groups=self.config.MODEL.RESNETS.NUM_GROUPS,
            width_per_group=self.config.MODEL.RESNETS.WIDTH_PER_GROUP,
            stride_in_1x1=self.config.MODEL.RESNETS.STRIDE_IN_1X1,
            stride_init=None,
            res2_out_channels=self.config.MODEL.RESNETS.RES2_OUT_CHANNELS,
            dilation=self.config.MODEL.RESNETS.RES5_DILATION
        )
        self.head = PoolHead(fastercnn_config, image_height, image_width)

        isize, nz, nc, ngf, ngpu, n_extra_layers = 64, 2048, 3, 128, 1, 0
        self.decoder = Decoder(isize, nz, nc, ngf, ngpu, n_extra_layers)

    def forward(self, feats, bboxes):
        # bboxes = BoxList(bboxes, (self.image_height, self.image_width), mode="xyxy")
        # x = self.pooler([feats, ], [bboxes, ])
        x =self.head(feats, bboxes)
        x = x.unsqueeze(2).unsqueeze(3)
        x = self.decoder(x)

        return x