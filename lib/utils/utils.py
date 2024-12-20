import matplotlib.pyplot as plt
import numpy as np
from torchvision.utils import make_grid
import torch


def mixup_batch(images, boxes, labels, alpha=1.0, mixup_fraction=1):
    batch_size = len(images)
    num_mix = int(batch_size * mixup_fraction)

    if num_mix == 0:
        return images, boxes, labels


    lambdas = np.random.beta(alpha, alpha, size=num_mix)
    lambdas = np.maximum(lambdas, 1 - lambdas)
    mix_indices = torch.randperm(batch_size)[:num_mix]

    shuffled_indices = torch.randperm(batch_size)
    shuffled_images = images[shuffled_indices]
    shuffled_boxes = [boxes[i].clone() for i in shuffled_indices]
    shuffled_labels = [labels[i].clone() for i in shuffled_indices]

    mixed_images = images.clone()
    mixed_boxes = []
    mixed_labels = []

    mix_count = 0
    for i in range(batch_size):
        if i in mix_indices:
            lam = lambdas[mix_count]
            mixed_img = images[i] * lam + shuffled_images[i] * (1-lam)
            mixed_images[i] = mixed_img
            mix_count += 1

            if boxes[i].numel() == 0 and shuffled_boxes[i].numel() == 0:
                mixed_boxes.append(torch.empty((0, 4), dtype=boxes[i].dtype).to(boxes[i].device))
                mixed_labels.append(torch.empty((0,), dtype=labels[i].dtype).to(labels[i].device))
            elif boxes[i].numel() == 0:
                mixed_boxes.append(shuffled_boxes[i])
                mixed_labels.append(shuffled_labels[i])
            elif shuffled_boxes[i].numel() == 0:
                mixed_boxes.append(boxes[i])
                mixed_labels.append(labels[i])
            else:
                mixed_boxes.append(torch.cat([boxes[i], shuffled_boxes[i]], dim=0))
                mixed_labels.append(torch.cat([labels[i], shuffled_labels[i]], dim=0))
        else:
            mixed_boxes.append(boxes[i].clone())
            mixed_labels.append(labels[i].clone())

    return mixed_images, mixed_boxes, mixed_labels

def imshow(img):
    npimg = img.numpy()
    if npimg.min() < 0 or npimg.max() > 1:
        npimg = (npimg - npimg.min()) / (npimg.max() - npimg.min())

    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.axis('off')
    plt.show()

def show_batch(dataloader, n=4):
    dataiter = iter(dataloader)
    images, labels, labels_rot = next(dataiter)
    imshow(make_grid(images[:n]))

def show_batch_cutmix(dataloader, n=4, alpha=0.5):
    dataiter = iter(dataloader)
    images, labels, labels_rot = next(dataiter)
    #images = images.float()
    images, labels, labels_rot = mixup_batch(images, labels, labels_rot, alpha=alpha)

    imshow(make_grid(images.int()[:n]))
