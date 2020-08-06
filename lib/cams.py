import os
import cv2
import numpy as np
import torch


def resize_cam(cam, size=(224, 224)):
    cam = cv2.resize(cam, (size[0], size[1]), interpolation=cv2.INTER_LINEAR)
    cam = cam - cam.min()
    cam = cam / cam.max()
    return cam


def blend_cam(image, cam):
    cam = (cam * 255.).astype(np.uint8)
    heatmap = cv2.applyColorMap(cam, cv2.COLORMAP_JET)
    blend = image * 0.5 + heatmap * 0.5

    return blend, heatmap


def tensor2image(input):
    image = input.numpy().transpose(0, 2, 3, 1)
    image = image[:, :, :, ::-1] * 255
    return image


def draw_cam(input, label, logits, cams, image_names, save_dir, cfg):
    """
    :param input: input tensors of the model
    :param label: class labels
    :param logits: classification scores
    :param cams: cam of all the classes
    :param image_names: names of images
    :param cfg: configurations
    :param save_dir: save cam
    :return: evaluate results
    """

    label = label.tolist()
    cls_scores = logits.tolist()

    batch = cams.shape[0]
    image = tensor2image(input)

    for b in range(batch):
        label_b = label[b]
        class_list = [x for x in range(len(label_b)) if label_b[x] == 1]

        for c in class_list:
            cam = cams[b, c, :, :]
            cam = cam.unsqueeze(0)

            cam = cam.detach().cpu().numpy().transpose(1, 2, 0)

            # Resize and Normalize CAM
            cam = resize_cam(cam)
            # # Get blended image
            blend, heatmap = blend_cam(image[b], cam)

            image_name = image_names[b]
            save_path = os.path.join(save_dir, image_name+'_'+str(c)+'.jpg')
            cv2.imwrite(save_path, blend)