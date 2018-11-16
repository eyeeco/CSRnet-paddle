# -*- coding: UTF-8 -*-
import numpy as np
import paddle.v2 as paddle
import logging
logging.basicConfig()
logger = logging.getLogger("paddle")
logger.setLevel(logging.INFO)

class CSRNet(object):
    def __init__(self,is_infer=False):
        super(CSRNet,self).__init__()
        # input and output layer
        self.image=paddle.layer.data(name='image', type=paddle.data_type.dense_vector(3*960*540), width=960, height=540, depth=3 )
        if not is_infer:
            self.label=paddle.layer.data(name='label',type=paddle.data_type.dense_vector(120*68), width=120, height=68, depth=1 )
        # frontend 
        # 定义五组卷积操作
        self.conv1 = self.conv_block(self.image, 64, 2, [0.3, 0], 3)
        self.conv2 = self.conv_block(self.conv1, 128, 2, [0.4, 0],64)
        self.conv3 = self.conv_block(self.conv2, 256, 3, [0.4, 0.4, 0],128)
        self.conv4 = self.conv_block(self.conv3, 512, 3, [0.4, 0.4, 0],256,with_poll=False)
        # backend
        self.backend_feat = [512, 512, 512, 256, 128, 64]
        self.backend = self.make_layers(self.backend_feat, self.conv4, in_channels=512, dilation=True)
        # output
        self.model =paddle.layer.img_conv(
            input=self.backend,
            num_filters=1,
            filter_size=1,
            name='output_layer',
            param_attr=paddle.attr.ParamAttr(
                initial_mean=0.0, initial_std=0.01),
            bias_attr=False)
        # cost
        if not is_infer:
            self.cost = paddle.layer.square_error_cost(input=self.model, label=self.label)
        
        
    def make_layers(self, cfg, input_layer,in_channels=3, dilation=False ):
        if dilation:
            d_rate = 2
        else:
            d_rate = 1
        _input_layer = input_layer
        for i, dim in enumerate(cfg):
            backend_layer = paddle.layer.img_conv(
                input=_input_layer,
                num_filters=dim,
                filter_size=3,
                stride=1,
                padding=d_rate,
                dilation=d_rate,
                act=paddle.activation.Relu(),
                #name='back_layer_%d'%i,
                param_attr=paddle.attr.ParamAttr(
                    initial_mean=0.0, initial_std=0.01),
                bias_attr=False)
            _input_layer=backend_layer
        return _input_layer    
    def conv_block(self, ipt, num_filter, groups, dropouts, num_channels=None ,with_poll=True):
        if not with_poll:
            pool_size = 1
            pool_stride = 1
        else:
            pool_size = 2
            pool_stride = 2
        return paddle.networks.img_conv_group(
            input=ipt,
            num_channels=num_channels,
            pool_size=pool_size,
            pool_stride=pool_stride,
            conv_num_filter=[num_filter] * groups,
            conv_filter_size=3,
            conv_act=paddle.activation.Relu(),
            conv_with_batchnorm=True,
            conv_batchnorm_drop_rate=dropouts,
            pool_type=paddle.pooling.Max())