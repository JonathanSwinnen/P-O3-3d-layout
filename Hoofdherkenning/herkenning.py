# Sample code from the TorchVision 0.3 Object Detection Finetuning Tutorial
# http://pytorch.org/tutorials/intermediate/torchvision_tutorial.html

import torch
import os

import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

# errors are fine, importing local files in ./pytorch_files/
from pytorch_files.engine import train_one_epoch, evaluate
import pytorch_files.utils as utils
import PO3_dataset
import plot_losses


def build_model(num_classes):
    # get moddel Resnet
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

def main():
    # train on the GPU or on the CPU, if a GPU is not available
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    if torch.cuda.is_available():
        print('Train on GPU.')
    else:
        print('Train on CPU.')

    paths = ('./adjusted_data/apart_0/', './adjusted_data/apart_1/',
            './adjusted_data/meer_pers_0/', './adjusted_data/meer_pers_1/',
            './adjusted_data/videodata_0/', './adjusted_data/videodata_1/',
            './adjusted_data/zittend_0/', './adjusted_data/zittend_1/')

    # debug/testing
    #paths = ('./adjusted_data/zittend_0/', './adjusted_data/zittend_1/')

    # our dataset has two classes only - background and heads
    num_classes = 2

    # use our dataset and defined transformations
    dataset = PO3_dataset.PO3Dataset_Training(paths, PO3_dataset.get_transform(train=True))
    dataset_test = PO3_dataset.PO3Dataset_Training(paths, PO3_dataset.get_transform(train=False))

    # split the dataset in train and test set randomly
    indices = torch.randperm(len(dataset)).tolist()
    dataset = torch.utils.data.Subset(dataset, indices[:-50])
    dataset_test = torch.utils.data.Subset(dataset_test, indices[-50:])

    # define training and validation data loaders
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=2, shuffle=True, num_workers=4,
        collate_fn=utils.collate_fn)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, shuffle=False, num_workers=4,
        collate_fn=utils.collate_fn)

    # get the model using our helper function
    model = build_model(num_classes)
    model.to(device)

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    # and a learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    MODEL_PATH = "./saved_models/PO3_v3/"
    try:
        os.mkdir(MODEL_PATH)
    except FileExistsError:
        # directory already exists
        pass

    tr_loss = []
    val_score = []

    # let's train it for a few epochs
    num_epochs = 6

    for epoch in range(num_epochs):
        # train for one epoch, printing every 10 iterations
        training_data = train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=20)
        print("Training Loss: ", str(training_data.loss)[:7])
        tr_loss.append(float(str(training_data.loss)[:7]))
        # update the learning rate
        lr_scheduler.step()
        # evaluate on the test dataset
        _, epoch_val_score = evaluate(model, data_loader_test, device=device)
        print("Validation score:", epoch_val_score)

        # save current version of the model
        print("Saved the model at:", os.path.join(MODEL_PATH, "training_" + str(epoch) + ".pth"))
        torch.save(model, os.path.join(MODEL_PATH, "training_" + str(epoch) + ".pth"))

    plot_losses.plt_losses(num_epochs, tr_loss, val_score, os.path.join(MODEL_PATH, "losses.png"))
    print(tr_loss, val_score)

    print("That's it! The best modelversion is: ", val_score.index(max(val_score)) + 1)
    print("Path: " + os.path.join(MODEL_PATH, "training_" + str(val_score.index(max(val_score)) + 1) + ".pth"))


if __name__ == "__main__":
    main()

