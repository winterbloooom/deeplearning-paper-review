import matplotlib.pyplot as plt
import numpy as np


def img_show(img, text=None):
    img_np = img.numpy()[0]  
    # img.numpy() shape: (1, 3, 227, 227) -> (batch, channel, W, H)
    # -> for one image, use img.numpy()[0]

    if text is not None:
        plt.title(text, loc="left", pad=20)
    plt.imshow(np.transpose(img_np, (1, 2, 0)))
    plt.show()


def get_lr(optimizer):
    for param_gruop in optimizer.param_groups:
        return param_gruop["lr"]
