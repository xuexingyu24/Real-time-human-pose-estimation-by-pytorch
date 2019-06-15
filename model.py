import torch
from collections import OrderedDict
import torch.nn as nn

############# CMU Model ###############

def make_layers(block, no_relu_layers):
    layers = []
    for layer_name, v in block.items():
        if 'pool' in layer_name:
            layer = nn.MaxPool2d(kernel_size=v[0], stride=v[1],
                                    padding=v[2])
            layers.append((layer_name, layer))
        else:
            conv2d = nn.Conv2d(in_channels=v[0], out_channels=v[1],
                               kernel_size=v[2], stride=v[3],
                               padding=v[4])
            layers.append((layer_name, conv2d))
            if layer_name not in no_relu_layers:
                layers.append(('relu_'+layer_name, nn.ReLU(inplace=True)))

    return nn.Sequential(OrderedDict(layers))

class bodypose_model(nn.Module):
    def __init__(self):
        super(bodypose_model, self).__init__()

        # these layers have no relu layer
        no_relu_layers = ['conv5_5_CPM_L1', 'conv5_5_CPM_L2', 'Mconv7_stage2_L1',\
                          'Mconv7_stage2_L2', 'Mconv7_stage3_L1', 'Mconv7_stage3_L2',\
                          'Mconv7_stage4_L1', 'Mconv7_stage4_L2', 'Mconv7_stage5_L1',\
                          'Mconv7_stage5_L2', 'Mconv7_stage6_L1', 'Mconv7_stage6_L1']
        blocks = {}
        block0 = OrderedDict({'conv1_1': [3, 64, 3, 1, 1],
                  'conv1_2': [64, 64, 3, 1, 1],
                  'pool1_stage1': [2, 2, 0],
                  'conv2_1': [64, 128, 3, 1, 1],
                  'conv2_2': [128, 128, 3, 1, 1],
                  'pool2_stage1': [2, 2, 0],
                  'conv3_1': [128, 256, 3, 1, 1],
                  'conv3_2': [256, 256, 3, 1, 1],
                  'conv3_3': [256, 256, 3, 1, 1],
                  'conv3_4': [256, 256, 3, 1, 1],
                  'pool3_stage1': [2, 2, 0],
                  'conv4_1': [256, 512, 3, 1, 1],
                  'conv4_2': [512, 512, 3, 1, 1],
                  'conv4_3_CPM': [512, 256, 3, 1, 1],
                  'conv4_4_CPM': [256, 128, 3, 1, 1]})


        # Stage 1
        block1_1 = OrderedDict({'conv5_1_CPM_L1': [128, 128, 3, 1, 1],
                    'conv5_2_CPM_L1': [128, 128, 3, 1, 1],
                    'conv5_3_CPM_L1': [128, 128, 3, 1, 1],
                    'conv5_4_CPM_L1': [128, 512, 1, 1, 0],
                    'conv5_5_CPM_L1': [512, 38, 1, 1, 0]})

        block1_2 = OrderedDict({'conv5_1_CPM_L2': [128, 128, 3, 1, 1],
                    'conv5_2_CPM_L2': [128, 128, 3, 1, 1],
                    'conv5_3_CPM_L2': [128, 128, 3, 1, 1],
                    'conv5_4_CPM_L2': [128, 512, 1, 1, 0],
                    'conv5_5_CPM_L2': [512, 19, 1, 1, 0]})
        blocks['block1_1'] = block1_1
        blocks['block1_2'] = block1_2

        self.model0 = make_layers(block0, no_relu_layers)

        # Stages 2 - 6
        for i in range(2, 7):
            blocks['block%d_1' % i] = OrderedDict({
                'Mconv1_stage%d_L1' % i: [185, 128, 7, 1, 3],
                'Mconv2_stage%d_L1' % i: [128, 128, 7, 1, 3],
                'Mconv3_stage%d_L1' % i: [128, 128, 7, 1, 3],
                'Mconv4_stage%d_L1' % i: [128, 128, 7, 1, 3],
                'Mconv5_stage%d_L1' % i: [128, 128, 7, 1, 3],
                'Mconv6_stage%d_L1' % i: [128, 128, 1, 1, 0],
                'Mconv7_stage%d_L1' % i: [128, 38, 1, 1, 0]})

            blocks['block%d_2' % i] = OrderedDict({
                'Mconv1_stage%d_L2' % i: [185, 128, 7, 1, 3],
                'Mconv2_stage%d_L2' % i: [128, 128, 7, 1, 3],
                'Mconv3_stage%d_L2' % i: [128, 128, 7, 1, 3],
                'Mconv4_stage%d_L2' % i: [128, 128, 7, 1, 3],
                'Mconv5_stage%d_L2' % i: [128, 128, 7, 1, 3],
                'Mconv6_stage%d_L2' % i: [128, 128, 1, 1, 0],
                'Mconv7_stage%d_L2' % i: [128, 19, 1, 1, 0]})

        for k in blocks.keys():
            blocks[k] = make_layers(blocks[k], no_relu_layers)

        self.model1_1 = blocks['block1_1']
        self.model2_1 = blocks['block2_1']
        self.model3_1 = blocks['block3_1']
        self.model4_1 = blocks['block4_1']
        self.model5_1 = blocks['block5_1']
        self.model6_1 = blocks['block6_1']

        self.model1_2 = blocks['block1_2']
        self.model2_2 = blocks['block2_2']
        self.model3_2 = blocks['block3_2']
        self.model4_2 = blocks['block4_2']
        self.model5_2 = blocks['block5_2']
        self.model6_2 = blocks['block6_2']


    def forward(self, x):

        out1 = self.model0(x)

        out1_1 = self.model1_1(out1)
        out1_2 = self.model1_2(out1)
        out2 = torch.cat([out1_1, out1_2, out1], 1)

        out2_1 = self.model2_1(out2)
        out2_2 = self.model2_2(out2)
        out3 = torch.cat([out2_1, out2_2, out1], 1)

        out3_1 = self.model3_1(out3)
        out3_2 = self.model3_2(out3)
        out4 = torch.cat([out3_1, out3_2, out1], 1)

        out4_1 = self.model4_1(out4)
        out4_2 = self.model4_2(out4)
        out5 = torch.cat([out4_1, out4_2, out1], 1)

        out5_1 = self.model5_1(out5)
        out5_2 = self.model5_2(out5)
        out6 = torch.cat([out5_1, out5_2, out1], 1)

        out6_1 = self.model6_1(out6)
        out6_2 = self.model6_2(out6)

        return out6_1, out6_2

################### Mobilenet ########################

def conv(in_channels, out_channels, kernel_size=3, padding=1, bn=True, dilation=1, stride=1, relu=True, bias=True):
    modules = [nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, bias=bias)]
    if bn:
        modules.append(nn.BatchNorm2d(out_channels))
    if relu:
        modules.append(nn.ReLU(inplace=True))
    return nn.Sequential(*modules)


def conv_dw(in_channels, out_channels, kernel_size=3, padding=1, stride=1, dilation=1):
    return nn.Sequential(
        nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, dilation=dilation, groups=in_channels, bias=False),
        nn.BatchNorm2d(in_channels),
        nn.ReLU(inplace=True),

        nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )


def conv_dw_no_bn(in_channels, out_channels, kernel_size=3, padding=1, stride=1, dilation=1):
    return nn.Sequential(
        nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, dilation=dilation, groups=in_channels, bias=False),
        nn.ELU(inplace=True),

        nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False),
        nn.ELU(inplace=True),
    )

class Cpm(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.align = conv(in_channels, out_channels, kernel_size=1, padding=0, bn=False)
        self.trunk = nn.Sequential(
            conv_dw_no_bn(out_channels, out_channels),
            conv_dw_no_bn(out_channels, out_channels),
            conv_dw_no_bn(out_channels, out_channels)
        )
        self.conv = conv(out_channels, out_channels, bn=False)

    def forward(self, x):
        x = self.align(x)
        x = self.conv(x + self.trunk(x))
        return x


class InitialStage(nn.Module):
    def __init__(self, num_channels, num_heatmaps, num_pafs):
        super().__init__()
        self.trunk = nn.Sequential(
            conv(num_channels, num_channels, bn=False),
            conv(num_channels, num_channels, bn=False),
            conv(num_channels, num_channels, bn=False)
        )
        self.heatmaps = nn.Sequential(
            conv(num_channels, 512, kernel_size=1, padding=0, bn=False),
            conv(512, num_heatmaps, kernel_size=1, padding=0, bn=False, relu=False)
        )
        self.pafs = nn.Sequential(
            conv(num_channels, 512, kernel_size=1, padding=0, bn=False),
            conv(512, num_pafs, kernel_size=1, padding=0, bn=False, relu=False)
        )

    def forward(self, x):
        trunk_features = self.trunk(x)
        heatmaps = self.heatmaps(trunk_features)
        pafs = self.pafs(trunk_features)
        return [heatmaps, pafs]


class RefinementStageBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.initial = conv(in_channels, out_channels, kernel_size=1, padding=0, bn=False)
        self.trunk = nn.Sequential(
            conv(out_channels, out_channels),
            conv(out_channels, out_channels, dilation=2, padding=2)
        )

    def forward(self, x):
        initial_features = self.initial(x)
        trunk_features = self.trunk(initial_features)
        return initial_features + trunk_features


class RefinementStage(nn.Module):
    def __init__(self, in_channels, out_channels, num_heatmaps, num_pafs):
        super().__init__()
        self.trunk = nn.Sequential(
            RefinementStageBlock(in_channels, out_channels),
            RefinementStageBlock(out_channels, out_channels),
            RefinementStageBlock(out_channels, out_channels),
            RefinementStageBlock(out_channels, out_channels),
            RefinementStageBlock(out_channels, out_channels)
        )
        self.heatmaps = nn.Sequential(
            conv(out_channels, out_channels, kernel_size=1, padding=0, bn=False),
            conv(out_channels, num_heatmaps, kernel_size=1, padding=0, bn=False, relu=False)
        )
        self.pafs = nn.Sequential(
            conv(out_channels, out_channels, kernel_size=1, padding=0, bn=False),
            conv(out_channels, num_pafs, kernel_size=1, padding=0, bn=False, relu=False)
        )

    def forward(self, x):
        trunk_features = self.trunk(x)
        heatmaps = self.heatmaps(trunk_features)
        pafs = self.pafs(trunk_features)
        return [heatmaps, pafs]


class PoseEstimationWithMobileNet(nn.Module):
    def __init__(self, num_refinement_stages=1, num_channels=128, num_heatmaps=19, num_pafs=38):
        super().__init__()
        self.model = nn.Sequential(
            conv(     3,  32, stride=2, bias=False),
            conv_dw( 32,  64),
            conv_dw( 64, 128, stride=2),
            conv_dw(128, 128),
            conv_dw(128, 256, stride=2),
            conv_dw(256, 256),
            conv_dw(256, 512),  # conv4_2
            conv_dw(512, 512, dilation=2, padding=2),
            conv_dw(512, 512),
            conv_dw(512, 512),
            conv_dw(512, 512),
            conv_dw(512, 512)   # conv5_5
        )
        self.cpm = Cpm(512, num_channels)

        self.initial_stage = InitialStage(num_channels, num_heatmaps, num_pafs)
        self.refinement_stages = nn.ModuleList()
        for idx in range(num_refinement_stages):
            self.refinement_stages.append(RefinementStage(num_channels + num_heatmaps + num_pafs, num_channels,
                                                          num_heatmaps, num_pafs))

    def forward(self, x):
        backbone_features = self.model(x)
        backbone_features = self.cpm(backbone_features)

        stages_output = self.initial_stage(backbone_features)
        for refinement_stage in self.refinement_stages:
            stages_output.extend(
                refinement_stage(torch.cat([backbone_features, stages_output[-2], stages_output[-1]], dim=1)))

        return stages_output

if __name__ == "__main__":

    import time 
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#    device = torch.device("cpu")
    input = torch.Tensor(2, 3, 368, 368).to(device)

    model_CMU = bodypose_model().to(device)
    model_CMU.load_state_dict(torch.load('weights/bodypose_model'))
    model_CMU.eval()
    
    model_Mobilenet = PoseEstimationWithMobileNet().to(device)
    model_Mobilenet.load_state_dict(torch.load('weights/MobileNet_bodypose_model'))
    model_Mobilenet.eval()
    
    since = time.time()
    
    PAF_CMU, Heatmap_CMU = model_CMU(input)
    print('CMU PAF shape and Heatmap shape', PAF_CMU.shape, Heatmap_CMU.shape)
    t1 = time.time()
    print('CMU Inference time is {:2.3f} seconds'.format(t1 - since))
    
    stages_output= model_Mobilenet(input)
    PAF_Mobilenet, Heatmap_Mobilenet = stages_output[-1], stages_output[-2]
    print('Mobilenet PAF shape and Heatmap shape', PAF_Mobilenet.shape, Heatmap_Mobilenet.shape)
    print('Mobilenet Inference time is {:2.3f} seconds'.format(time.time() - t1))
    


