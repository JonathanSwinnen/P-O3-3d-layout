import matplotlib.pyplot as plt
import time


def plt_losses(epochs, tr_loss, val_score, store_path="./saved_models/losses.png"):

    # add values
    plt.plot(epochs, tr_loss, 'r', label='Training losses')
    plt.plot(epochs, val_score, 'g', label='Validation score')

    # labels
    plt.xlabel('Epochs')
    plt.ylabel('Value')
    plt.title('Training parameters in function of epochs')
    plt.legend()

    # save plot as .png
    plt.savefig(store_path)