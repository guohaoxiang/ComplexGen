# Copyright (c) Chris Choy (chrischoy@ai.stanford.edu).
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
# of the Software, and to permit persons to whom the Software is furnished to do
# so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# Please cite "4D Spatio-Temporal ConvNets: Minkowski Convolutional Neural
# Networks", CVPR'19 (https://arxiv.org/abs/1904.08755) if you use any part
# of the code.
import torch
import torch.nn as nn
from torch.optim import SGD
from torch.nn import Module

import MinkowskiEngine as ME
from MinkowskiEngine.modules.resnet_block import BasicBlock, Bottleneck
from MinkowskiEngine import MinkowskiNonlinearity
from MinkowskiSparseTensor import SparseTensor
from MinkowskiCommon import (
    MinkowskiModuleBase,
    get_minkowski_function,
)

class MinkowskiLeakyReLU(MinkowskiNonlinearity.MinkowskiNonlinearityBase):
    MODULE = torch.nn.LeakyReLU

class MinkowskiInstanceNormGuo_backup(MinkowskiModuleBase):
    #this is an wrong implimentation, update 0928, layer norm, not instance norm
    r"""A instance normalization layer for a sparse tensor.

    """

    def __init__(self, num_features):
        r"""
        Args:

            num_features (int): the dimension of the input feautres.

            mode (GlobalPoolingModel, optional): The internal global pooling computation mode.
        """
        Module.__init__(self)
        self.num_features = num_features
        self.weight = nn.Parameter(torch.ones(1, num_features))
        self.bias = nn.Parameter(torch.zeros(1, num_features))
        self.reset_parameters()
        # self.inst_norm = MinkowskiInstanceNormFunctionGuo()

    def __repr__(self):
        s = f"(nchannels={self.num_features})"
        return self.__class__.__name__ + s

    def reset_parameters(self):
        self.weight.data.fill_(1)
        self.bias.data.zero_()

    def forward(self, input: SparseTensor):
        assert isinstance(input, SparseTensor)

        # output = self.inst_norm.apply(
        #     input.F, input.coordinate_map_key, None, input.coordinate_manager
        # )


        # list_of_feats = input.decomposed_features
        # list_of_coords, list_of_feats = input.decomposed_coordinates_and_features
        
        # output_l = []
        # for f in list_of_feats:
        #     m = f.mean()
        #     var = torch.var(f) + eps
        #     std = var.sqrt()
        #     f = (f - m) / std
        #     f = f * self.weight + self.bias
        #     output_l.append(f)

        # coords, feats = ME.utils.sparse_collate(
        # coords=list_of_coords, feats=output_l)
        # output = ME.SparseTensor(coordinates=coords, features=feats)
        # return output
        # output = ME.SparseTensor(features=feats, coordinate_map_key=input.coordinate_map_key, coordinate_manager=input.coordinate_manager)
        # return output

        list_of_features = input.decomposed_features
        list_of_permutations = input.decomposition_permutations
        eps = 1e-5
        for f,inds in zip(list_of_features, list_of_permutations):
            # normalize f
            m = f.mean()
            var = torch.var(f) + eps
            std = var.sqrt()
            f = (f - m) / std
            f = f * self.weight + self.bias
            input.F[inds] = f
        return input


class MinkowskiInstanceNormGuo(MinkowskiModuleBase):
    r"""A instance normalization layer for a sparse tensor.

    """

    def __init__(self, num_features):
        r"""
        Args:

            num_features (int): the dimension of the input feautres.

            mode (GlobalPoolingModel, optional): The internal global pooling computation mode.
        """
        Module.__init__(self)
        self.num_features = num_features
        self.weight = nn.Parameter(torch.ones(1, num_features))
        self.bias = nn.Parameter(torch.zeros(1, num_features))
        self.reset_parameters()
        # self.inst_norm = MinkowskiInstanceNormFunctionGuo()

    def __repr__(self):
        s = f"(nchannels={self.num_features})"
        return self.__class__.__name__ + s

    def reset_parameters(self):
        self.weight.data.fill_(1)
        self.bias.data.zero_()

    def forward(self, input: SparseTensor):
        assert isinstance(input, SparseTensor)

        # output = self.inst_norm.apply(
        #     input.F, input.coordinate_map_key, None, input.coordinate_manager
        # )


        # list_of_feats = input.decomposed_features
        # list_of_coords, list_of_feats = input.decomposed_coordinates_and_features
        
        # output_l = []
        # for f in list_of_feats:
        #     m = f.mean()
        #     var = torch.var(f) + eps
        #     std = var.sqrt()
        #     f = (f - m) / std
        #     f = f * self.weight + self.bias
        #     output_l.append(f)

        # coords, feats = ME.utils.sparse_collate(
        # coords=list_of_coords, feats=output_l)
        # output = ME.SparseTensor(coordinates=coords, features=feats)
        # return output
        # output = ME.SparseTensor(features=feats, coordinate_map_key=input.coordinate_map_key, coordinate_manager=input.coordinate_manager)
        # return output

        list_of_features = input.decomposed_features
        list_of_permutations = input.decomposition_permutations
        eps = 1e-6
        for f,inds in zip(list_of_features, list_of_permutations):
            # normalize f
            m = f.mean(dim = 0) #64
            var = torch.var(f, dim = 0) + eps
            std = var.sqrt()
            f = (f - m) / std
            f = f * self.weight + self.bias
            input.F[inds] = f
        return input



class ResNetBase(nn.Module):
    BLOCK = None
    LAYERS = ()
    INIT_DIM = 64
    PLANES = (64, 128, 256, 512)

    def __init__(self, in_channels, out_channels, D=3):
        nn.Module.__init__(self)
        self.D = D
        assert self.BLOCK is not None

        self.network_initialization(in_channels, out_channels, D)
        self.weight_initialization()

    def network_initialization(self, in_channels, out_channels, D):
        
        self.inplanes = self.INIT_DIM
        self.conv1 = nn.Sequential(
            ME.MinkowskiConvolution(
                in_channels, self.inplanes, kernel_size=3, stride=1, dimension=D #stride was 2 here
            ),
            # MinkowskiInstanceNormGuo(self.inplanes),
            MinkowskiInstanceNormGuo_backup(self.inplanes), #modify 20210928
            MinkowskiLeakyReLU(inplace=True),
            ME.MinkowskiMaxPooling(kernel_size=2, stride=2, dimension=D),
        )

        self.layer1 = self._make_layer(
            self.BLOCK, self.PLANES[0], self.LAYERS[0], stride=1 #stride was 2
        )
        self.layer2 = self._make_layer(
            self.BLOCK, self.PLANES[1], self.LAYERS[1], stride=2
        )
        self.layer3 = self._make_layer(
            self.BLOCK, out_channels, self.LAYERS[2], stride=2 #self.PLANES[2]
        )


        
        # not downsample 20210510
        # self.layer3 = self._make_layer(
        #     self.BLOCK, out_channels, self.LAYERS[2], stride=1 #self.PLANES[2]
        # )
        '''
        self.layer4 = self._make_layer(
            self.BLOCK, self.PLANES[3], self.LAYERS[3], stride=2
        )

        self.conv5 = nn.Sequential(
            ME.MinkowskiDropout(),
            ME.MinkowskiConvolution(
                self.inplanes, self.inplanes, kernel_size=3, stride=3, dimension=D
            ),
            ME.MinkowskiInstanceNorm(self.inplanes),
            ME.MinkowskiGELU(),
        )

        self.glob_pool = ME.MinkowskiGlobalMaxPooling()

        self.final = ME.MinkowskiLinear(self.inplanes, out_channels, bias=True)
        '''

    def weight_initialization(self):
        for m in self.modules():
            if isinstance(m, ME.MinkowskiConvolution):
                ME.utils.kaiming_normal_(m.kernel, mode="fan_out", nonlinearity="relu")

            if isinstance(m, MinkowskiInstanceNormGuo_backup):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1, bn_momentum=0.1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                ME.MinkowskiConvolution(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    dimension=self.D,
                ),
                MinkowskiInstanceNormGuo_backup(planes * block.expansion),
            )
        layers = []
        layers.append(
            block(
                self.inplanes,
                planes,
                stride=stride,
                dilation=dilation,
                downsample=downsample,
                dimension=self.D,
            )
        )
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(
                block(
                    self.inplanes, planes, stride=1, dilation=dilation, dimension=self.D
                )
            )

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        # x = self.layer4(x)
        #x = self.layer4(x)
        #x = self.conv5(x)
        #x = self.glob_pool(x)
        return x#self.final(x)


class ResNet14(ResNetBase):
    BLOCK = BasicBlock
    LAYERS = (1, 1, 1, 1)


class ResNet18(ResNetBase):
    BLOCK = BasicBlock
    LAYERS = (2, 2, 2, 2)


class ResNet34(ResNetBase):
    BLOCK = BasicBlock
    LAYERS = (3, 4, 6, 3)


class BasicBlockOur(nn.Module):
    expansion = 1
    # NORM = MinkowskiInstanceNormGuo
    NORM = None

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 dilation=1,
                 downsample=None,
                 bn_momentum=0.1,
                 dimension=-1):
        super(BasicBlockOur, self).__init__()
        assert dimension > 0

        # self.conv1 = ME.MinkowskiConvolution(
        #     inplanes, planes, kernel_size=3, stride=stride, dilation=dilation, dimension=dimension)
        
        self.conv1 = ME.MinkowskiConvolution(
            inplanes, planes, kernel_size=3, stride=1, dilation=dilation, dimension=dimension)
        
        # self.norm1 = ME.MinkowskiBatchNorm(planes, momentum=bn_momentum)
        self.norm1 = self.NORM(planes)

        self.conv2 = ME.MinkowskiConvolution(
            planes, planes, kernel_size=3, stride=1, dilation=dilation, dimension=dimension)
        # self.norm2 = ME.MinkowskiBatchNorm(planes, momentum=bn_momentum)
        self.norm2 = self.NORM(planes)

        # self.relu = ME.MinkowskiReLU(inplace=True)
        self.relu = MinkowskiLeakyReLU(inplace=True)
        
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.norm2(out)

        if self.downsample is not None:
          residual = self.downsample(x)
        out += residual
        out = self.relu(out)

        return out

class BasicBlockConvTranspose(nn.Module):
    expansion = 1
    # NORM = MinkowskiInstanceNormGuo
    NORM = None

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 dilation=1,
                 downsample=None,
                 bn_momentum=0.1,
                 dimension=-1):
        super(BasicBlockOur, self).__init__()
        assert dimension > 0

        # self.conv1 = ME.MinkowskiConvolution(
        #     inplanes, planes, kernel_size=3, stride=stride, dilation=dilation, dimension=dimension)
        
        self.conv1 = ME.MinkowskiConvolution(
            inplanes, planes, kernel_size=3, stride=1, dilation=dilation, dimension=dimension)
        
        # self.norm1 = ME.MinkowskiBatchNorm(planes, momentum=bn_momentum)
        self.norm1 = self.NORM(planes)

        self.conv2 = ME.MinkowskiConvolution(
            planes, planes, kernel_size=3, stride=1, dilation=dilation, dimension=dimension)
        # self.norm2 = ME.MinkowskiBatchNorm(planes, momentum=bn_momentum)
        self.norm2 = self.NORM(planes)

        # self.relu = ME.MinkowskiReLU(inplace=True)
        self.relu = MinkowskiLeakyReLU(inplace=True)
        
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.norm2(out)

        if self.downsample is not None:
          residual = self.downsample(x)
        out += residual
        out = self.relu(out)

        return out

class BasicBlockOurBn(BasicBlockOur):
    NORM = ME.MinkowskiBatchNorm

class BasicBlockOurIn(BasicBlockOur):
    NORM = MinkowskiInstanceNormGuo

class ResNetBaseOur(nn.Module):
    BLOCK = None
    LAYERS = ()
    PLANES = (64, 128, 256, 512)
    NORM = MinkowskiInstanceNormGuo

    def __init__(self, in_channels, out_channels, D=3, flag_expand = False):
        nn.Module.__init__(self)
        self.D = D
        assert self.BLOCK is not None
        self.flag_expand = flag_expand
        self.network_initialization(in_channels, out_channels, D)
        self.weight_initialization()

    def network_initialization(self, in_channels, out_channels, D):
        
        if not self.flag_expand:
            self.conv1 = nn.Sequential(
                ME.MinkowskiConvolution(
                    in_channels, self.PLANES[0], kernel_size=3, stride=1, dimension=D #stride was 2 here
                ),
                self.NORM(self.PLANES[0]),
                MinkowskiLeakyReLU(inplace=True),
                # ME.MinkowskiMaxPooling(kernel_size=2, stride=2, dimension=D),   #TODO: do we need this pooling?
            )
        else:
            self.conv1 = nn.Sequential(
                ME.MinkowskiConvolution(
                    in_channels, self.PLANES[0], kernel_size=3, stride=1, dimension=D, expand_coordinates=True
                ),
                self.NORM(self.PLANES[0]),
                MinkowskiLeakyReLU(inplace=True),
                # ME.MinkowskiMaxPooling(kernel_size=2, stride=2, dimension=D),   #TODO: do we need this pooling?
            )

        self.layer1 = self._make_layer(
            self.BLOCK, self.PLANES[0], self.PLANES[1], self.LAYERS[0], stride=1 
        )
        self.layer2 = self._make_layer(
            self.BLOCK, self.PLANES[1], self.PLANES[2], self.LAYERS[1], stride=1
        )
        self.layer3 = self._make_layer(
            self.BLOCK, self.PLANES[2], out_channels, self.LAYERS[2], stride=1 #self.PLANES[2]
        )
        
        self.layer4 = self.BLOCK(
                out_channels,
                out_channels,
                stride=1,
                dimension=self.D,
            )


    def weight_initialization(self):
        for m in self.modules():
            if isinstance(m, ME.MinkowskiConvolution):
                ME.utils.kaiming_normal_(m.kernel, mode="fan_out", nonlinearity="relu")

            if isinstance(m, MinkowskiInstanceNormGuo):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, in_planes, out_planes, blocks, stride=1, dilation=1, bn_momentum=0.1):
            
        layers = []
        
        for i in range(0, blocks):  # residual blocks
            layers.append(
                block(
                    in_planes, in_planes, stride=1, dilation=dilation, dimension=self.D
                )
            )

        layers.append(
            nn.Sequential( # conv1x1 to increase feature size
                ME.MinkowskiConvolution(
                    in_planes,
                    out_planes,
                    kernel_size=1,
                    stride=stride,
                    dimension=self.D,
                ),
                self.NORM(out_planes),
                MinkowskiLeakyReLU(inplace=True), )
            )

        # pooling to reduce spatial resolution
        layers.append(ME.MinkowskiMaxPooling(kernel_size=2, stride=2, dimension=self.D))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x 

class ResNetBaseDecoderRec(nn.Module):
    BLOCK = None
    LAYERS = ()
    # PLANES = (64, 128, 256, 512)
    PLANES = (256, 128, 64)
    # NORM = MinkowskiInstanceNormGuo
    NORM = None

    def __init__(self, in_channels, out_channels, D=3):
        nn.Module.__init__(self)
        self.D = D
        assert self.BLOCK is not None

        self.network_initialization(in_channels, out_channels, D)
        self.weight_initialization()

    def network_initialization(self, in_channels, out_channels, D):
        
        self.layer1 = self.BLOCK(
                in_channels,
                in_channels,
                stride=1,
                dimension=self.D,
            )
        self.layer2 = self._make_layer(
            self.BLOCK, in_channels, self.PLANES[0], self.LAYERS[0], stride=1 
        )
        self.layer3 = self._make_layer(
            self.BLOCK, self.PLANES[0], self.PLANES[1], self.LAYERS[1], stride=1
        )
        self.layer4 = self._make_layer(
            self.BLOCK, self.PLANES[1], self.PLANES[2], self.LAYERS[2], stride=1 #self.PLANES[2]
        )

        self.conv1 = nn.Sequential(
            ME.MinkowskiConvolution(
                self.PLANES[2], out_channels, kernel_size=3, stride=1, dimension=D #stride was 2 here
            ),
            # MinkowskiInstanceNormGuo(out_channels),
            # MinkowskiLeakyReLU(inplace=True),
            # ME.MinkowskiMaxPooling(kernel_size=2, stride=2, dimension=D),   #TODO: do we need this pooling?
        )


    def weight_initialization(self):
        for m in self.modules():
            if isinstance(m, ME.MinkowskiConvolution):
                ME.utils.kaiming_normal_(m.kernel, mode="fan_out", nonlinearity="relu")

            if isinstance(m, MinkowskiInstanceNormGuo):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, in_planes, out_planes, blocks, stride=1, dilation=1, bn_momentum=0.1):
            
        layers = []

        # pooling to reduce spatial resolution
        # layers.append(ME.MinkowskiPoolingTranspose(kernel_size=2, stride=2, dimension=self.D))
        # layers.append(ME.MinkowskiGenerativeConvolutionTranspose(in_planes, in_planes, kernel_size=2, stride=2, dimension = self.D))
        layers.append(ME.MinkowskiConvolutionTranspose(in_planes, in_planes, kernel_size=2, stride=2, dimension = self.D))


        layers.append(
            nn.Sequential( # conv1x1 to increase feature size
                ME.MinkowskiConvolution(
                    in_planes,
                    out_planes,
                    kernel_size=1,
                    stride=stride,
                    dimension=self.D,
                ),
                self.NORM(out_planes),
                MinkowskiLeakyReLU(inplace=True), )
            )
        
        for i in range(0, blocks):  # residual blocks
            layers.append(
                block(
                    out_planes, out_planes, stride=1, dilation=dilation, dimension=self.D
                )
            )

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.conv1(x)
        return x 

class ResNetBaseDecoderOur(nn.Module):
    BLOCK = None
    LAYERS = ()
    # PLANES = (64, 128, 256, 512)
    PLANES = (256, 128, 64)
    # NORM = MinkowskiInstanceNormGuo
    NORM = None

    def __init__(self, in_channels, out_channels, D=3):
        nn.Module.__init__(self)
        self.D = D
        assert self.BLOCK is not None

        self.network_initialization(in_channels, out_channels, D)
        self.weight_initialization()

    def network_initialization(self, in_channels, out_channels, D):
        
        self.layer1 = self.BLOCK(
                in_channels,
                in_channels,
                stride=1,
                dimension=self.D,
            )
        self.layer2 = self._make_layer(
            self.BLOCK, in_channels, self.PLANES[0], self.LAYERS[0], stride=1 
        )
        self.layer3 = self._make_layer(
            self.BLOCK, self.PLANES[0], self.PLANES[1], self.LAYERS[1], stride=1
        )
        self.layer4 = self._make_layer(
            self.BLOCK, self.PLANES[1], self.PLANES[2], self.LAYERS[2], stride=1 #self.PLANES[2]
        )

        self.conv1 = nn.Sequential(
            ME.MinkowskiConvolution(
                self.PLANES[2], out_channels, kernel_size=3, stride=1, dimension=D #stride was 2 here
            ),
            # MinkowskiInstanceNormGuo(out_channels),
            # MinkowskiLeakyReLU(inplace=True),
            # ME.MinkowskiMaxPooling(kernel_size=2, stride=2, dimension=D),   #TODO: do we need this pooling?
        )


    def weight_initialization(self):
        for m in self.modules():
            if isinstance(m, ME.MinkowskiConvolution):
                ME.utils.kaiming_normal_(m.kernel, mode="fan_out", nonlinearity="relu")

            if isinstance(m, MinkowskiInstanceNormGuo):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, in_planes, out_planes, blocks, stride=1, dilation=1, bn_momentum=0.1):
            
        layers = []

        # pooling to reduce spatial resolution
        layers.append(ME.MinkowskiPoolingTranspose(kernel_size=2, stride=2, dimension=self.D))

        layers.append(
            nn.Sequential( # conv1x1 to increase feature size
                ME.MinkowskiConvolution(
                    in_planes,
                    out_planes,
                    kernel_size=1,
                    stride=stride,
                    dimension=self.D,
                ),
                self.NORM(out_planes),
                MinkowskiLeakyReLU(inplace=True), )
            )
        
        for i in range(0, blocks):  # residual blocks
            layers.append(
                block(
                    out_planes, out_planes, stride=1, dilation=dilation, dimension=self.D
                )
            )

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.conv1(x)
        return x 




class ResNetOur(ResNetBaseOur):
    BLOCK = BasicBlockOurIn
    NORM = MinkowskiInstanceNormGuo
    LAYERS = (3, 3, 3)

class ResNetDecoderOur(ResNetBaseDecoderOur):
    BLOCK = BasicBlockOurIn
    NORM = MinkowskiInstanceNormGuo
    LAYERS = (3, 3, 3)

class ResNetDecoderRecBn(ResNetBaseDecoderRec):
    BLOCK = BasicBlockOurBn
    NORM = ME.MinkowskiBatchNorm
    LAYERS = (3, 3, 3)

class ResNetOurBn(ResNetBaseOur):
    BLOCK = BasicBlockOurBn
    NORM = ME.MinkowskiBatchNorm
    LAYERS = (3, 3, 3)

class ResNetDecoderOurBn(ResNetBaseDecoderOur):
    BLOCK = BasicBlockOurBn
    NORM = ME.MinkowskiBatchNorm
    LAYERS = (3, 3, 3)

class ResNet50(ResNetBase):
    BLOCK = Bottleneck
    LAYERS = (3, 4, 6, 3)


class ResNet101(ResNetBase):
    BLOCK = Bottleneck
    LAYERS = (3, 4, 23, 3)


class ResFieldNetBase(ResNetBase):
    def network_initialization(self, in_channels, out_channels, D):
        field_ch = 32
        field_ch2 = 64
        self.field_network = nn.Sequential(
            ME.MinkowskiSinusoidal(in_channels, field_ch),
            ME.MinkowskiInstanceNorm(field_ch),
            ME.MinkowskiReLU(inplace=True),
            ME.MinkowskiLinear(field_ch, field_ch),
            ME.MinkowskiInstanceNorm(field_ch),
            ME.MinkowskiReLU(inplace=True),
            ME.MinkowskiToSparseTensor(),
        )
        self.field_network2 = nn.Sequential(
            ME.MinkowskiSinusoidal(field_ch + in_channels, field_ch2),
            ME.MinkowskiInstanceNorm(field_ch2),
            ME.MinkowskiReLU(inplace=True),
            ME.MinkowskiLinear(field_ch2, field_ch2),
            ME.MinkowskiInstanceNorm(field_ch2),
            ME.MinkowskiReLU(inplace=True),
            ME.MinkowskiToSparseTensor(),
        )

        ResNetBase.network_initialization(self, field_ch2, out_channels, D)

    def forward(self, x):
        otensor = self.field_network(x)
        otensor2 = self.field_network2(otensor.cat_slice(x))
        return ResNetBase.forward(self, otensor2)


class ResFieldNet14(ResFieldNetBase):
    BLOCK = BasicBlock
    LAYERS = (1, 1, 1, 1)


class ResFieldNet18(ResFieldNetBase):
    BLOCK = BasicBlock
    LAYERS = (2, 2, 2, 2)


class ResFieldNet34(ResFieldNetBase):
    BLOCK = BasicBlock
    LAYERS = (3, 4, 6, 3)


class ResFieldNet50(ResFieldNetBase):
    BLOCK = Bottleneck
    LAYERS = (3, 4, 6, 3)


class ResFieldNet101(ResFieldNetBase):
    BLOCK = Bottleneck
    LAYERS = (3, 4, 23, 3)


if __name__ == "__main__":
    # loss and network
    from tests.python.common import data_loader

    criterion = nn.CrossEntropyLoss()
    net = ResNet14(in_channels=3, out_channels=5, D=2)
    print(net)

    # a data loader must return a tuple of coords, features, and labels.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    net = net.to(device)
    optimizer = SGD(net.parameters(), lr=1e-2)

    for i in range(10):
        optimizer.zero_grad()

        # Get new data
        coords, feat, label = data_loader()
        input = ME.SparseTensor(feat, coords=coords).to(device)
        label = label.to(device)

        # Forward
        output = net(input)

        # Loss
        loss = criterion(output.F, label)
        print("Iteration: ", i, ", Loss: ", loss.item())

        # Gradient
        loss.backward()
        optimizer.step()

    # Saving and loading a network
    torch.save(net.state_dict(), "test.pth")
    net.load_state_dict(torch.load("test.pth"))
