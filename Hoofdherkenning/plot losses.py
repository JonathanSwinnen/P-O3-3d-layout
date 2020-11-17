import matplotlib.pyplot as plt
import time

epochs = [i for i in range(35)]
losses = [0.2229, 0.2125, 0.1642, 0.1053, 0.1009, 0.1192, 0.0807, 0.0788, 0.0962, 0.1265,
          0.1108,  0.1296, 0.0775, 0.0918, 0.0952, 0.1092, 0.1220, 0.1100, 0.1023, 0.0938,
          0.0901, 0.0836, 0.1245, 0.0941, 0.0950, 0.0640, 0.1015, 0.1063, 0.0782, 0.1059,
          0.1155, 0.0876, 0.0865, 0.0941, 0.1282]

plt.plot(epochs, losses)
plt.xlabel('Epochs')
plt.ylabel('loss')
plt.title('Training loss in function of epochs')
plt.show()