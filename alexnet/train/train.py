import torch
import torch.optim as optim
from torchsummary import summary
from utils.tools import *

from train.loss import STL10Loss


def train_model(
    _model,
    device,
    train_loader,
    batch,
    n_classes,
    in_channel,
    in_width,
    in_height,
    _epoch,
    output_dir,
    torch_writer,
):
    print("\n" + "=" * 50 + " Train Model " + "=" * 50)

    model = _model(
        batch=batch,
        n_classes=n_classes,
        in_channel=in_channel,
        in_width=in_width,
        in_height=in_height,
        is_train=True,
    )
    model.to(device)
    model.train()

    # check model architecture
    print(model)
    summary(model, input_size=(3, 227, 227), batch_size=32, device=device.type)

    # check parameters initialization
    # for param in model.parameters():
    #     print(param)
    #     break

    optimizer = optim.SGD(
        model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0005
    )
    # optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

    scheduler = optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=30, gamma=0.1)
    
    criterion = STL10Loss(device=device)

    epoch = _epoch
    iter = 0

    for e in range(epoch):
        total_loss = 0
        correct = 0
        img_num = 0

        for i, batch in enumerate(train_loader):
            img = batch[0]
            img = img.to(device)
            label = batch[1]
            label = label.to(device)

            out = model(img)  # shape: (batch, n_classes)
            _, pred = torch.max(out, 1) #(input Tensor, dim) -> Tuple(max, max_indices)
            # print("pred:", pred)
            # print("label: ", label)

            loss_val = criterion(out, label)
            loss_val.backward()

            optimizer.step()
            optimizer.zero_grad()

            total_loss += loss_val.item()
            correct += torch.sum(pred == label)
            img_num += len(label)

            if iter % 10 == 0:
                print(
                    "{:>2} / {:>2} epoch ({:>3.0f} %) {:>3} iter \t loss: {:>.8f} \t accuracy: {:>4} / {:>4} ({:>4.2f} %)".format(
                        e,
                        epoch,
                        100.0 * i / len(train_loader),
                        iter,
                        loss_val.item(),
                        correct.item(),
                        img_num,
                        100.0 * correct.item() / img_num,
                    )
                )

                torch_writer.add_scalar("lr", get_lr(optimizer), iter)
                torch_writer.add_scalar("total_loss", loss_val, iter)
                torch_writer.add_scalar("accuracy", 100.0 * correct / img_num, iter)

            iter += 1

        scheduler.step()

        mean_loss = total_loss / i
        mean_accuracy = correct / img_num
        print(
            "-> {:>2} / {:>2} epoch \t mean loss: {} \t mean accuracy: {}".format(
                e, epoch, mean_loss, mean_accuracy
            )
        )

        torch.save(model.state_dict(), output_dir + "/model_epoch" + str(e) + ".pt")

    print("=" * 50 + " Train End " + "=" * 50)
