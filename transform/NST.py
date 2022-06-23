import utils.utils as utils
import torch
from torch.optim import LBFGS
from torch.autograd import Variable
import numpy as np
import os
import argparse


def neural_style_transfer(content_img_name, style_img_name, is_preserve_color=False, cw=3e8, sw=1e7, tw=1e4, init_method='content'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = os.path.join(os.path.dirname(__file__), 'data')
    content_img_dir = os.path.join(data_dir, 'content_images')
    style_img_dir = os.path.join(data_dir, 'style_images')
    output_img_dir = os.path.join(data_dir, 'output_images')
    content_img_path = os.path.join(content_img_dir, content_img_name)
    style_img_path = os.path.join(style_img_dir, style_img_name)
    output_img_name = content_img_name + '+' + style_img_name
    output_img_path = os.path.join(output_img_dir, output_img_name)

    if torch.cuda.is_available():
        print("cuda is available")

    content_img = utils.prepare_img(content_img_path, device, padding=True, is_gauss=True)
    style_img = utils.prepare_img(style_img_path, device)

    if init_method == 'random':
        gaussian_noise_img = np.random.normal(loc=0, scale=90., size=content_img.shape).astype(np.float32)
        init_img = torch.from_numpy(gaussian_noise_img).float().to(device)
    elif init_method == 'content':
        init_img = content_img

    current_img = Variable(init_img, requires_grad=True)
    copy_content_img = current_img.squeeze(axis=0).cpu().detach().numpy()
    copy_content_img = np.transpose(copy_content_img)
    copy_content_img = np.swapaxes(copy_content_img, 0, 1)
    neural_net = utils.prepare_model(device)
    target_content_features = utils.content_feature(content_img, neural_net)
    target_style_features = utils.style_feature(style_img, neural_net)

    num_of_iterations = {
        "lbfgs": 500,
        "adam": 3000,
    }

    optimizer = LBFGS((current_img,), max_iter=num_of_iterations['lbfgs'], line_search_fn='strong_wolfe')
    cnt = 0

    def closure():
        nonlocal cnt
        if torch.is_grad_enabled():
            optimizer.zero_grad()
        total_loss = utils.loss(neural_net, target_content_features, target_style_features, current_img, cw, sw, tw)
        if total_loss.requires_grad:
            total_loss.backward()
        with torch.no_grad():
            print(f'L-BFGS | iteration: {cnt:03}, total loss={total_loss.item():12.4f}')
            cnt += 1
        return total_loss

    optimizer.step(closure)

    utils.save(current_img, output_img_path, preserve_color=is_preserve_color, origin_content_img=copy_content_img)
    return output_img_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--content_img_name", type=str, help="content image name", default='default_content_image.jpg')
    parser.add_argument("--style_img_name", type=str, help="style image name", default='default_style_image.jpg')
    parser.add_argument("--content_weight", type=float, help="weight factor for content loss", default=3e8)
    parser.add_argument("--style_weight", type=float, help="weight factor for style loss", default=1e7)
    parser.add_argument("--tv_weight", type=float, help="weight factor for total variation loss", default=1e4)
    parser.add_argument("--init_method", type=str, choices=['random', 'content', 'style'], default='content')
    parser.add_argument("--is_preserve_color", type=bool, choices=[True, False], default=False)
    args = parser.parse_args()

    optimization_config = dict()
    for arg in vars(args):
        optimization_config[arg] = getattr(args, arg)
    neural_style_transfer(optimization_config["content_img_name"], optimization_config["style_img_name"],
                          optimization_config["is_preserve_color"], optimization_config["content_weight"],
                          optimization_config["style_weight"], optimization_config["tv_weight"],
                          optimization_config["init_method"])
