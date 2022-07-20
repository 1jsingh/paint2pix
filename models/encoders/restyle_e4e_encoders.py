from enum import Enum
from torch import nn
from torch.nn import Conv2d, BatchNorm2d, PReLU, Sequential, Module
from torchvision.models import resnet34

from models.encoders.helpers import get_blocks, bottleneck_IR, bottleneck_IR_SE
from models.encoders.map2style import GradualStyleBlock

from models.stylegan2.model import EqualLinear

class ProgressiveStage(Enum):
    WTraining = 0
    Delta1Training = 1
    Delta2Training = 2
    Delta3Training = 3
    Delta4Training = 4
    Delta5Training = 5
    Delta6Training = 6
    Delta7Training = 7
    Delta8Training = 8
    Delta9Training = 9
    Delta10Training = 10
    Delta11Training = 11
    Delta12Training = 12
    Delta13Training = 13
    Delta14Training = 14
    Delta15Training = 15
    Delta16Training = 16
    Delta17Training = 17
    Inference = 18


class ProgressiveBackboneEncoder(Module):
    """
    Paint2pix uses a Restyle-like architecture for building the canvas and identity encoder. 
    This is a combined class which can be used as either the canvas or identity encoders 
    depending on the input arguements 'is_canvas_encoder=False'
    """
    def __init__(self, num_layers, mode='ir', n_styles=18, opts=None, is_canvas_encoder=False):
        super(ProgressiveBackboneEncoder, self).__init__()
        assert num_layers in [50, 100, 152], 'num_layers should be 50,100, or 152'
        assert mode in ['ir', 'ir_se'], 'mode should be ir or ir_se'
        blocks = get_blocks(num_layers)
        if mode == 'ir':
            unit_module = bottleneck_IR
        elif mode == 'ir_se':
            unit_module = bottleneck_IR_SE

        self.input_layer = Sequential(Conv2d(opts.input_nc, 64, (3, 3), 1, 1, bias=False),
                                      BatchNorm2d(64),
                                      PReLU(64))

        self.is_canvas_encoder = opts.is_canvas_encoder
        if self.is_canvas_encoder:
            self.input_layer_updated_canvas = Sequential(Conv2d(opts.input_nc, 64, (3, 3), 1, 1, bias=False),
                                      BatchNorm2d(64),
                                      PReLU(64))

        modules = []
        for block in blocks:
            for bottleneck in block:
                modules.append(unit_module(bottleneck.in_channel,
                                           bottleneck.depth,
                                           bottleneck.stride))
        self.body = Sequential(*modules)

        self.styles = nn.ModuleList()
        self.style_count = n_styles
        for i in range(self.style_count):
            style = GradualStyleBlock(512, 512, 16)
            self.styles.append(style)
        self.progressive_stage = ProgressiveStage.Inference
        
        self.id_layers = nn.ModuleList()
        for i in range(self.style_count):
            id_layer = EqualLinear(512, 512, lr_mul=1)
            self.id_layers.append(id_layer)

    def get_deltas_starting_dimensions(self):
        ''' Get a list of the initial dimension of every delta from which it is applied '''
        return list(range(self.style_count))  # Each dimension has a delta applied to

    def set_progressive_stage(self, new_stage: ProgressiveStage):
        #  In this encoder we train all the pyramid (At least as a first stage experiment
        self.progressive_stage = new_stage
        print('Changed progressive stage to: ', new_stage)

    def forward(self, x_, target_id_feat=None, get_multiple_codes=False):
        x = self.input_layer(x_)
        x = self.body(x)

        # get initial w0 from first map2style layer
        w0 = self.styles[0](x)
        if target_id_feat is not None:
            w0 += self.id_layers[0](target_id_feat)
        w = w0.repeat(self.style_count, 1, 1).permute(1, 0, 2)

        # learn the deltas up to the current stage
        stage = self.progressive_stage.value
        for i in range(1, min(stage + 1, self.style_count)):
            delta_i = self.styles[i](x) 
            if target_id_feat is not None:
                delta_i += self.id_layers[i](target_id_feat)
            w[:, i] += delta_i

        if self.is_canvas_encoder and get_multiple_codes:
            w_canvas0 = w
            x = self.input_layer_updated_canvas(x_)
            x = self.body(x)

            # get initial w0 from first map2style layer
            w0 = self.styles[0](x)
            if target_id_feat is not None:
                w0 += self.id_layers[0](target_id_feat)
            w = w0.repeat(self.style_count, 1, 1).permute(1, 0, 2)

            # learn the deltas up to the current stage
            stage = self.progressive_stage.value
            for i in range(1, min(stage + 1, self.style_count)):
                delta_i = self.styles[i](x) 
                if target_id_feat is not None:
                    delta_i += self.id_layers[i](target_id_feat)
                w[:, i] += delta_i
            w_canvas1 = w
            return w_canvas0, w_canvas1
        else:
            return w