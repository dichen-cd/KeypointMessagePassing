from torchreid.models import *
from torchreid.models import __model_factory

from .joints_gait import JointsGait
from .deeper_gcn import DeeperGCN
from .vg_interact import VisualGaitJointNet
from .keypoint_message_passing import KeypointMessagePassing, KeypointMessagePassingRes50, \
                                      KeypointNoPassing, KeypointMessagePassingWithAttention, \
                                      KeypointMessageDistillation

__model_factory['gcn'] = JointsGait
__model_factory['deepergcn'] = DeeperGCN
__model_factory['vginteract'] = VisualGaitJointNet
__model_factory['kmpnet'] = KeypointMessagePassing
__model_factory['kmpnetres50'] = KeypointMessagePassingRes50

__model_factory['knpnet'] = KeypointNoPassing
__model_factory['kmpnet_att'] = KeypointMessagePassingWithAttention
__model_factory['kmdnet'] = KeypointMessageDistillation


def build_model(
    name, num_classes, loss='softmax', pretrained=True, use_gpu=True, **kwargs
):
    """A function wrapper for building a model.
    Args:
        name (str): model name.
        num_classes (int): number of training identities.
        loss (str, optional): loss function to optimize the model. Currently
            supports "softmax" and "triplet". Default is "softmax".
        pretrained (bool, optional): whether to load ImageNet-pretrained weights.
            Default is True.
        use_gpu (bool, optional): whether to use gpu. Default is True.
    Returns:
        nn.Module
    Examples::
        >>> from torchreid import models
        >>> model = models.build_model('resnet50', 751, loss='softmax')
    """
    avai_models = list(__model_factory.keys())
    if name not in avai_models:
        raise KeyError(
            'Unknown model: {}. Must be one of {}'.format(name, avai_models)
        )
    return __model_factory[name](
        num_classes=num_classes,
        loss=loss,
        pretrained=pretrained,
        use_gpu=use_gpu,
        **kwargs
    )
