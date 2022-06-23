import cv2
import numpy as np
import torch
from torchvision import models
from torchvision import transforms


class Vgg19(torch.nn.Module):
    def __init__(self):
        super().__init__()
        pretrained_models = models.vgg19(pretrained=True)
        '''改用relu层而非论文中层作为内容以及风格'''
        self.offset = 1
        self.content_layers = 'conv4_2'
        self.style_layers = ['relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'relu5_1']
        pretrained_features = pretrained_models.features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        self.slice6 = torch.nn.Sequential()
        for x in range(1 + self.offset):
            self.slice1.add_module(str(x), pretrained_features[x])
        for x in range(1 + self.offset, 6 + self.offset):
            self.slice2.add_module(str(x), pretrained_features[x])
        for x in range(6 + self.offset, 11 + self.offset):
            self.slice3.add_module(str(x), pretrained_features[x])
        for x in range(11 + self.offset, 20 + self.offset):
            self.slice4.add_module(str(x), pretrained_features[x])
        for x in range(20 + self.offset, 22):
            self.slice5.add_module(str(x), pretrained_features[x])
        for x in range(22, 29 + +self.offset):
            self.slice6.add_module(str(x), pretrained_features[x])
        for para in self.parameters():
            para.requires_grad = False

    def forward(self, x):
        x = self.slice1(x)
        layer1_1 = x
        x = self.slice2(x)
        layer2_1 = x
        x = self.slice3(x)
        layer3_1 = x
        x = self.slice4(x)
        layer4_1 = x
        x = self.slice5(x)
        conv4_2 = x
        x = self.slice6(x)
        layer5_1 = x
        out = (layer1_1, layer2_1, layer3_1, layer4_1, conv4_2, layer5_1)
        return out


def prepare_model(device):
    model = Vgg19()
    return model.to(device).eval()


MEAN_255 = [123.675, 116.28, 103.53]
VAR_255 = [1, 1, 1]


def gauss(img):
    img = cv2.GaussianBlur(img, (5, 5), 0)
    img = cv2.GaussianBlur(img, (5, 5), 0)
    img = cv2.GaussianBlur(img, (5, 5), 0)
    return img


def prepare_img(img_path, device, padding=False, is_resize=False, target_width=0, target_height=0, is_gauss=True):
    img = cv2.imread(img_path)
    img = img[:, :, ::-1]

    current_height, current_width = img.shape[:2]
    new_height = 400
    new_width = int(current_width * (new_height / current_height))
    img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
    if is_resize:
        img = cv2.resize(img, [target_width, target_height], interpolation=cv2.INTER_CUBIC)
    img = img.astype(np.float32)
    img /= 255.0

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255)),
        transforms.Normalize(mean=MEAN_255, std=VAR_255)])

    if padding:
        img = cv2.copyMakeBorder(img, 50, 50, 50, 50, cv2.BORDER_DEFAULT)

    if is_gauss:
        img = gauss(img)

    img = transform(img).to(device).unsqueeze(0)
    return img


def gram_matrix(x, should_normalize=True):
    (batch_size, filter_num, height, width) = x.size()
    features = x.view(batch_size, filter_num, height * width)
    features_t = features.transpose(1, 2)
    gram = features.bmm(features_t)
    if should_normalize:
        gram /= filter_num * height * width
    return gram


def content_feature(content_img, neural_net):
    content = neural_net(content_img)[4].squeeze(axis=0)
    return content


def style_feature(style_img, neural_net):
    layers = neural_net(style_img)
    style = [gram_matrix(x) for cnt, x in enumerate(layers) if cnt in [0, 1, 2, 3, 5]]
    return style


def loss(neural_net, target_content_feature, target_style_feature, current_img, cw, sw, tw):
    current_layers = neural_net(current_img)
    current_content_feature = current_layers[4].squeeze(0)
    current_style_feature = [gram_matrix(x) for cnt, x in enumerate(current_layers) if cnt in [0, 1, 2, 3, 5]]

    content_loss = torch.nn.MSELoss(reduction='mean')(target_content_feature, current_content_feature)

    style_loss = 0.0
    for gram_gt, gram_hat in zip(target_style_feature, current_style_feature):
        style_loss += torch.nn.MSELoss(reduction='sum')(gram_gt[0], gram_hat[0])
    style_loss /= len(target_style_feature)

    tv_loss = torch.sum(torch.abs(current_img[:, :, :, :-1] - current_img[:, :, :, 1:])) + torch.sum(torch.abs(current_img[:, :, :-1, :] - current_img[:, :, 1:, :]))

    total_loss = cw * content_loss + sw * style_loss + tw * tv_loss
    return total_loss


def preprocess(img):
    img_pre = np.copy(img)
    img_pre = img_pre[..., ::-1]
    img_pre -= np.array(MEAN_255).reshape(1, 1, 3)
    return img_pre


def postprocess(img):
    img_post = np.copy(img)
    img_post += np.array(MEAN_255).reshape((1, 1, 3))
    img_post = np.clip(img_post, 0, 255).astype('uint8')
    img_post = img_post[..., ::-1]
    return img_post


def convert_to_origin(content_img, stylized_img):
    stylized_img = postprocess(stylized_img)

    content_img = postprocess(content_img)

    cvt_type = cv2.COLOR_BGR2YUV
    inv_cvt_type = cv2.COLOR_YUV2BGR
    content_cvt = cv2.cvtColor(np.float32(content_img), cvt_type)
    stylized_cvt = cv2.cvtColor(np.float32(stylized_img), cvt_type)
    c1, _, _ = cv2.split(stylized_cvt)
    _, c2, c3 = cv2.split(content_cvt)
    merged = cv2.merge((c1, c2, c3))
    dst = cv2.cvtColor(merged, inv_cvt_type).astype(np.float32)

    dst = preprocess(dst)
    return dst


def save(current_img, path, preserve_color=False, origin_content_img=None):
    out_img = current_img.squeeze(axis=0).to('cpu').detach().numpy()
    out_img = np.moveaxis(out_img, 0, 2)
    dump_img = np.copy(out_img)
    if preserve_color:
        dump_img = convert_to_origin(origin_content_img, dump_img)
    dump_img += np.array(MEAN_255).reshape((1, 1, 3))
    dump_img = np.clip(dump_img, 0, 255).astype('uint8')
    dump_img = dump_img[50:-50, 50: -50]
    cv2.imwrite(path, dump_img[:, :, ::-1])
