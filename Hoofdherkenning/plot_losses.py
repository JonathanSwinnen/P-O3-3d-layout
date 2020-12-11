import matplotlib.pyplot as plt
import time
import pickle

# This import registers the 3D projection, but is otherwise unused.
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np


def plt_losses(epochs, tr_loss, val_score, store_path="./saved_models/losses.png"):
    tr_losses = tr_loss.copy()
    val_scores = val_score.copy()
    epochs = range(epochs)

    # add values
    plt.plot(epochs, tr_losses, 'r', label='Training losses')
    plt.plot(epochs, val_scores, 'g', label='Validation score')

    # labels
    plt.xlabel('Epochs')
    plt.ylabel('Value')
    plt.title('Training parameters in function of epochs')
    plt.legend()

    # save plot as .png
    plt.savefig(store_path)

def plt_custom_scores():
    path = "saved_models/PO3_v6/"
    data = []
    for i in range(31):
         file = open(path + 'training_' + str(i) + '.pckl', 'rb')
         # dump information to that file
         data.append(pickle.load(file))
         # close the file
         file.close()
   #for i in range(81):
   #    file = open(path[:-2] + '10/data_' + str(i) + '.pckl', 'rb')
   #    # dump information to that file
   #    data.append(pickle.load(file))
   #    # close the file
   #    file.close()
    # training_losses = data[49][0]

    # recall = [(i[2][1][4] + i[2][2][4])/(i[2][1][3] + i[2][2][3]) for i in data]
    # precision = [(i[2][1][4] + i[2][2][4]) / (i[2][1][4] + i[2][1][5] + i[2][2][4] + i[2][2][5]) for i in data]

   # iou = [i[2][2][6] for i in data]
    # dist = [i[2][2][7] for i in data]
   #training_loss = [i[0][-1:][0] for i in data]
    recall_h_t = [i[0][1][4] / i[0][1][3] for i in data]
    recall_m_t = [i[0][2][4] / i[0][2][3] for i in data]
    recall_h_g = [i[1][1][4] / i[1][1][3] for i in data]
    recall_m_g = [i[1][2][4] / i[1][2][3] for i in data]


   #training_l_adj = [sum(training_loss[i:i+7])/7 for i in range(131-6)]
   #recall_h_t_adj = [sum(recall_h_t[i:i+7])/7 for i in range(131-6)]
   #recall_m_t_adj = [sum(recall_m_t[i:i+7])/7 for i in range(131-6)]
   #recall_h_g_adj = [sum(recall_h_g[i:i+7])/7 for i in range(131-6)]
   #recall_m_g_adj = [sum(recall_m_g[i:i+7])/7 for i in range(131-6)]


    # plt.plot(range(50), training_losses, label='Training loss')
    # plt.plot(range(50), recall, label='Recall')
    # plt.plot(range(50), precision, label='Precision')
    # plt.plot(range(50), iou, label='IoU')
  # plt.plot(range(51), training_loss, color="b", alpha=0.3)
   #plt.plot(range(51), recall_h_t, color="orange", alpha=0.3)
   #plt.plot(range(51), recall_m_t, color="r", alpha=0.3)
   #plt.plot(range(51), recall_h_g, color="g", alpha=0.3)
   #plt.plot(range(51), recall_m_g, color="purple", alpha=0.3)


    plt.plot(range(31), recall_h_t, color="orange", label="recall heads test set")
    plt.plot(range(31), recall_m_t, color="r", label="recall masks test set")
    plt.plot(range(31), recall_h_g, color="g", label="recall heads generalisation set")
    plt.plot(range(31), recall_m_g, color="purple", label="recall masks generalisation set")
#
 # plt.plot(range(6, 131), training_l_adj, color="b", label='training loss (7 epoch average)')
 # plt.plot(range(6, 131), recall_h_t_adj, color="orange", label='recall heads test set (7 epoch average)')
 # plt.plot(range(6, 131), recall_m_t_adj, color="r", label='recall masks test set (7 epoch average)')
 # plt.plot(range(6, 131), recall_h_g_adj, color="g", label='recall heads generalisation set (7 epoch average)')
 # plt.plot(range(6, 131), recall_m_g_adj, color="purple", label='recall masks generalisation set (7 epoch average)')


    #plt.plot(range(50), dist, label='Distance')
    plt.title("Recall pretrained")
    plt.legend()
    plt.ylim(ymin=0)
    plt.show()

def results():
    file = open("./saved_models/PO3_v8/data_49.pckl", 'rb')
    data = pickle.load(file)
    file.close()

    recall_h_t = data[1][1][4] / data[1][1][3] * 100
    recall_m_t = data[1][2][4] / data[1][2][3] * 100
    recall_h_g = data[2][1][4] / data[2][1][3] * 100
    recall_m_g = data[2][2][4] / data[2][2][3] * 100

    precision_h_t = data[1][1][4] / (data[1][1][4] + data[1][1][5]) * 100
    precision_m_t = data[1][2][4] / (data[1][2][4] + data[1][2][5]) * 100
    precision_h_g = data[2][1][4] / (data[2][1][4] + data[2][1][5]) * 100
    precision_m_g = data[2][2][4] / (data[2][2][4] + data[2][2][5]) * 100


    iou_h_t = data[1][1][6] * 100
    iou_m_t = data[1][2][6] * 100
    iou_h_g = data[2][1][6] * 100
    iou_m_g = data[2][2][6] * 100

    dist_h_t = data[1][1][7]
    dist_m_t = data[1][2][7]
    dist_h_g = data[2][1][7]
    dist_m_g = data[2][2][7]

    print("recall test:   ", recall_h_t , recall_m_t)
    print("recall gen:    ", recall_h_g, recall_m_g)

    print("precision test:", precision_h_t, precision_m_t)
    print("precision gen: ", precision_h_g, precision_m_g)

    print("iou test:      ", iou_h_t, iou_m_t)
    print("iou gen:       ", iou_h_g, iou_m_g)

    print("distance test: ", dist_h_t, dist_m_t)
    print("distance gen:  ", dist_h_g, dist_m_g)

def scores_f():
    path = "./saved_models/PO3_v9/"
    scores = []
    for i in range(50):
        file = open(path + 'data_' + str(i) + '.pckl', 'rb')
        scores.append(pickle.load(file)[1][1][0])
        file.close()

    adjusted_scores = []
    for i in scores:
        temp = dict()
        for j in range(101):
            temp[round(j/100, 2)] = 0

        for key, value in i.items():
            temp[round(key, 2)] += value
        adjusted_scores.append(temp)

    adjusted_scores_f = []
    for epoch in adjusted_scores:
        temp = []
        for i in range(0,101):
            temp.append(epoch[round(i/100, 2)])
        adjusted_scores_f.append(np.array(temp[:91]))
    print(adjusted_scores_f)
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    # Plot the surface
    # .
    # Make data.
    X = np.arange(0, .91, 0.01)
    Y = np.arange(0, 50, 1)
    X, Y = np.meshgrid(X, Y)

    Z = np.meshgrid(np.array(adjusted_scores_f))



    surf = ax.plot_surface(X, Y, np.array(adjusted_scores_f), cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)

    # Customize the z axis.
    ax.set_zlim(0, 25)
    ax.view_init(azim=-105)
    #ax.zaxis.set_major_locator(LinearLocator(10))
    #ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    ax.set_zlabel("#occurrences")
    ax.set_xlabel("score")
    ax.set_ylabel("epoch")
    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)
    ax.set_title("Score during epochs masks generalisation: [0, 0.9]")
    plt.show()






if __name__ == "__main__":
    # training_loss = [0.3035, 0.2611, 0.2024, 0.1709, 0.2103, 0.1699, 0.1452, 0.2076, 0.1784, 0.163, 0.1284, 0.1346,
    #                  0.1558, 0.1098, 0.1412, 0.1568, 0.1641, 0.1701, 0.1511, 0.182, 0.171, 0.1756, 0.1072, 0.1585,
    #                  0.1223, 0.1053, 0.1415, 0.1452, 0.147, 0.173, 0.1577, 0.1445, 0.1317, 0.1407, 0.1614, 0.1502,
    #                  0.1815, 0.1542, 0.173, 0.1359, 0.1174, 0.15, 0.1187, 0.1983, 0.1286, 0.1891, 0.1543, 0.1303,
    #                  0.1348, 0.1291, 0.1534, 0.1368, 0.1493, 0.2527, 0.1229, 0.1303, 0.1604, 0.1699, 0.1429, 0.1197,
    #                  0.1457, 0.1309, 0.118, 0.0998, 0.1043, 0.1829, 0.1282, 0.0964, 0.1496, 0.1275, 0.1771, 0.1378,
    #                  0.1673, 0.1715, 0.139, 0.1414, 0.1263, 0.1163, 0.1423, 0.1408, 0.089, 0.1473, 0.1701, 0.1248,
    #                  0.1464, 0.1795, 0.139, 0.1743, 0.1872, 0.182, 0.156, 0.1398, 0.1866, 0.1328, 0.1605, 0.1613,
    #                  0.1486, 0.108, 0.1413, 0.1563]
    # val_score = [0.30823312274047304, 0.3250738607377422, 0.5202089651506774, 0.6582906063722105, 0.7377542908094368,
    #            0.7384109703861937, 0.7481786341083293, 0.7369726263746923, 0.7593080279778461, 0.7624964787035572,
    #            0.7652129007845508, 0.764260112022867, 0.7642358863840297, 0.7642214134031412, 0.7660393082365697,
    #            0.7660383460473041, 0.7660391342883207, 0.7660389171571148, 0.7660389208063787, 0.7660388776234218,
    #            0.7660389122914295, 0.7660389110750082, 0.7660388812726858, 0.7660388928286883, 0.7660389208063787,
    #            0.7660389025600589, 0.766038893436899, 0.7660388946533203, 0.7660388946533203, 0.7660388946533203,
    #            0.7660388946533203, 0.7660388946533203, 0.7660388946533203, 0.7660388946533203, 0.7660388946533203,
    #            0.7660388946533203, 0.7660388946533203, 0.7660388946533203, 0.7660388946533203, 0.7660388946533203,
    #            0.7660388946533203, 0.7660388946533203, 0.7660388946533203, 0.7660388946533203, 0.7660388946533203,
    #            0.7660388946533203, 0.7660388946533203, 0.7660388946533203, 0.7660388946533203, 0.7660388946533203,
    #            0.7660388946533203, 0.7660388946533203, 0.7660388946533203, 0.7660388946533203, 0.7660388946533203,
    #            0.7660388946533203, 0.7660388946533203, 0.7660388946533203, 0.7660388946533203, 0.7660388946533203,
    #            0.7660388946533203, 0.7660388946533203, 0.7660388946533203, 0.7660388946533203, 0.7660388946533203,
    #            0.7660388946533203, 0.7660388946533203, 0.7660388946533203, 0.7660388946533203, 0.7660388946533203,
    #            0.7660388946533203, 0.7660388946533203, 0.7660388946533203, 0.7660388946533203, 0.7660388946533203,
    #            0.7660388946533203, 0.7660388946533203, 0.7660388946533203, 0.7660388946533203, 0.7660388946533203,
    #            0.7660388946533203, 0.7660388946533203, 0.7660388946533203, 0.7660388946533203, 0.7660388946533203,
    #            0.7660388946533203, 0.7660388946533203, 0.7660388946533203, 0.7660388946533203, 0.7660388946533203,
    #            0.7660388946533203, 0.7660388946533203, 0.7660388946533203, 0.7660388946533203, 0.7660388946533203,
    #            0.7660388946533203, 0.7660388946533203, 0.7660388946533203, 0.7660388946533203, 0.7660388946533203]

    # plt_losses(100, training_loss, val_score, "./saved_models/PO3_v3/losses.png")
    # plt_custom_scores()
    results()
    #scores_f()