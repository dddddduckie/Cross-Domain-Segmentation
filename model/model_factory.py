from model.deeplab_v2 import DeebLab

name_map = {
    "DeepLab": DeebLab
}


def create_model(num_classes, backbone='resnet', name='DeepLab'):
    support_act = ['DeepLab', 'PSPNet']
    assert support_act.__contains__(name)

    model = name_map[name](num_classes=num_classes, backbone=backbone)
    # model = bn_to_synbn(model)

    return model
