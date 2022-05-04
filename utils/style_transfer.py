import torch
import torch.nn.functional as F


def gram_matrix(input):
    """
    compute the gram matrix.
    """
    b, c, h, w = input.size()
    feature = input.view(b * c, h * w)
    g = torch.mm(feature, feature.t())
    return torch.div(g, h * c * h * w)


def content_loss(input, target):
    """
    compute the content loss, i.e., mean squared loss.
    """
    return F.mse_loss(input, target)


def style_loss_gram(input, target):
    """
    compute the style loss using gram matrix
    """
    gram_input = gram_matrix(input)
    gram_target = gram_matrix(target)
    return F.mse_loss(gram_input, gram_target)


def style_loss_in(input, target):
    """
    compute the style loss using instance normalization
    """
    input_mean, input_std = mean_std(input)
    target_mean, target_std = mean_std(target)
    return F.mse_loss(input_mean, target_mean) + F.mse_loss(input_std, target_std)


def mean_std(feature_map, eps=1e-5):
    """
    compute mean and std of the given feature map.
    eps is set to avoid divide by zero
    """
    b, c = feature_map.size()[0:2]
    mean = feature_map.view(b, c, -1).mean(dim=2).view(b, c, 1, 1)
    var = feature_map.view(b, c, -1).var(dim=2) + eps
    std = var.sqrt().view(b, c, 1, 1)
    return mean, std


def adin(content, style):
    """
    adaptive instance normalization
    """
    assert (content.size()[0:2] == style.size()[0:2])
    content_mean, content_std = mean_std(content)
    style_mean, style_std = mean_std(style)
    size = content.size()
    stylized_content = (content - content_mean.expand(size)) / content_std.expand(size)
    stylized_content = stylized_content * style_std.expand(size) + style_mean.expand(size)
    return stylized_content
