import torch
from utils.tools import *


def test_model(
    _model,
    device,
    test_loader,
    batch,
    n_classes,
    in_channel,
    in_width,
    in_height,
    _checkpoint,
):
    print("\n" + "=" * 50 + " Test Start " + "=" * 50)
    model = _model(
        batch=batch,
        n_classes=n_classes,
        in_channel=in_channel,
        in_width=in_width,
        in_height=in_height,
        is_train=False,
    )
    checkpoint = torch.load(_checkpoint)
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()

    correct = 0

    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            img = batch[0]
            label = batch[1]
            img = img.to(device)
            label = label.to(device)

            out = model(img)
            correct += torch.sum(out == label)

            text = "GT: " + str(label.item()) + " \t Output: " + str(out.item())
            # img_show(img=img.cpu(), text=text)
            print("#" + str(i) + "\t" + text)
            
        accuracy = 100. * correct / len(test_loader)
        print("Accuracy: {}".format(accuracy))

    print("=" * 50 + "Test End" + "=" * 50)
