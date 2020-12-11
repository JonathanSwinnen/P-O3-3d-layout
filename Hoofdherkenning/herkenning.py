# Sample code from the TorchVision 0.3 Object Detection Finetuning Tutorial
# http://pytorch.org/tutorials/intermediate/torchvision_tutorial.html

import torch
import os
import pickle
import sys

import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

# errors are fine, importing local files in ./pytorch_files/
from pytorch_files.engine import train_one_epoch, evaluate
import pytorch_files.utils as utils
import PO3_dataset
import plot_losses



def build_model(num_classes):
    # get moddel Resnet
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    model.load_state_dict(torch.load("./saved_models/PO3_v10/training_23.pt"))
    return model


def main():
    # train on the GPU or on the CPU, if a GPU is not available
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    if torch.cuda.is_available():
        print('Train on GPU.')
    else:
        print('Train on CPU.')

    # debug/testing
    #paths_training = ('./raw_data/TAFELS_0/', './raw_data/TAFELS_1/')

    paths_training = ('./raw_data/apart_0/', './raw_data/apart_1/',
                      './raw_data/zittend_0/', './raw_data/zittend_1/',
                      './raw_data/TAFELS_0/', './raw_data/TAFELS_1/',
                      './raw_data/two_0/', './raw_data/two_1/')

    paths_testing = ('./raw_data/meer_pers_0/','./raw_data/meer_pers_1/')

    paths_generalisation = ('./raw_data/TA_0/','./raw_data/TA_1/')

    # debug/testing
    # paths = ('./adjusted_data/zittend_0/', './adjusted_data/zittend_1/')

    # our dataset has two classes only - background and heads
    num_classes = 3

    # use our dataset and defined transformations
    dataset = PO3_dataset.PO3Dataset(paths_training, PO3_dataset.get_transform(train=True),
                                     has_sub_maps=True, ann_path="./raw_data/clean_ann_scaled.pckl")
    dataset_test = PO3_dataset.PO3Dataset(paths_testing, PO3_dataset.get_transform(train=False),
                                          has_sub_maps=True, ann_path="./raw_data/clean_ann_scaled.pckl")
    dataset_generalisation = PO3_dataset.PO3Dataset(paths_generalisation, PO3_dataset.get_transform(train=False),
                                          has_sub_maps=True, ann_path="./raw_data/clean_ann_scaled.pckl")

    # split the dataset in train and test set randomly
    indices = torch.randperm(len(dataset)).tolist()
    indices_test = torch.randperm(len(dataset_test)).tolist()
    indices_generalistation = torch.randperm(len(dataset_generalisation)).tolist()

    dataset = torch.utils.data.Subset(dataset, indices)
    dataset_test = torch.utils.data.Subset(dataset_test, indices_test)
    dataset_generalistation = torch.utils.data.Subset(dataset_generalisation, indices_generalistation)

    # define training and validation data loaders
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=2, shuffle=True, num_workers=4,
        collate_fn=utils.collate_fn)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, shuffle=False, num_workers=4,
        collate_fn=utils.collate_fn)

    data_loader_generalisation = torch.utils.data.DataLoader(
        dataset_generalistation, batch_size=1, shuffle=False, num_workers=4,
        collate_fn=utils.collate_fn)

    # get the model using our helper function
    model = build_model(num_classes)
    model.to(device)

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    # and a learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    MODEL_PATH = "./saved_models/PO3_v10/"
    try:
        os.mkdir(MODEL_PATH)
    except FileExistsError:
        # directory already exists
        pass

    tr_loss = []

    # let's train it for a few epochs
    num_epochs = 100

    for epoch in range(num_epochs):
        # train for one epoch, printing every 10 iterations
        training_data = train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=20)
        print("Training Loss: ", str(training_data.loss)[:7])
        tr_loss.append(float(str(training_data.loss)[:7]))
        # update the learning rate
        # lr_scheduler.step()
        # evaluate on the test dataset

        _, epoch_validation = evaluate(model, data_loader_test, device=device)
        _, epoch_generalisation = evaluate(model, data_loader_generalisation, device=device)

        print("Results: ")
        print("Heads testset:", epoch_validation[0][1:])
        print("Masks testset:", epoch_validation[1][1:])
        print("Heads generalisationset:", epoch_generalisation[0][1:])
        print("Masks generalisationset:", epoch_generalisation[1][1:])

        with open(os.path.join(MODEL_PATH, ('data_' + str(epoch + 24) + '.pckl')), 'wb') as f:
            pickle.dump((tr_loss, epoch_validation, epoch_generalisation), f, protocol=pickle.HIGHEST_PROTOCOL)

        # save current version of the model
        print("Saved the model at:", os.path.join(MODEL_PATH, "training_" + str(epoch + 24) + ".pt"))
        torch.save(model.state_dict(), os.path.join(MODEL_PATH, "training_" + str(epoch + 24) + ".pt"))

    print(tr_loss)

    print("That's it! Everything saved at:", MODEL_PATH)

    # print("Path: " + os.path.join(MODEL_PATH, "training_" + str(val_score.index(max(val_score)) + 1) + ".pth"))
    #plot_losses.plt_losses(num_epochs, tr_loss, val_score, os.path.join(MODEL_PATH, "losses.png"))


if __name__ == "__main__":
    main()

