import matplotlib.pyplot as plt
import time
import pickle

# This import registers the 3D projection, but is otherwise unused.
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
import matplotlib.pyplot as plt
import pickle
import numpy as np
import statistics

"""
This file was used to generate the plots in the final report.
This file is a draft version, which changes depending on what needs to be plotted.
"""


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
    file = open("./saved_models/PO3_v5/training_49.pckl", 'rb')
    data = pickle.load(file)
    file.close()

    recall_h_t = data[0][1][4] / data[0][1][3] * 100
    recall_m_t = data[0][2][4] / data[0][2][3] * 100
    recall_h_g = data[1][1][4] / data[1][1][3] * 100
    recall_m_g = data[1][2][4] / data[1][2][3] * 100

    precision_h_t = data[0][1][4] / (data[0][1][4] + data[0][1][5]) * 100
    precision_m_t = data[0][2][4] / (data[0][2][4] + data[0][2][5]) * 100
    precision_h_g = data[1][1][4] / (data[1][1][4] + data[1][1][5]) * 100
    precision_m_g = data[1][2][4] / (data[1][2][4] + data[1][2][5]) * 100


    iou_h_t = data[0][1][6] * 100
    iou_m_t = data[0][2][6] * 100
    iou_h_g = data[1][1][6] * 100
    iou_m_g = data[1][2][6] * 100

    dist_h_t = data[0][1][7]
    dist_m_t = data[0][2][7]
    dist_h_g = data[1][1][7]
    dist_m_g = data[1][2][7]

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

def time_plot():
    # data = [2.255601406097412, 0.2424759864807129, 0.23484158515930176, 0.23417282104492188, 0.24936223030090332, 0.23297381401062012, 0.23639321327209473, 0.2325141429901123, 0.24195122718811035, 0.23105406761169434, 0.24011611938476562, 0.24669647216796875, 0.23869562149047852, 0.23560357093811035, 0.23497962951660156, 0.23806977272033691, 0.24617576599121094, 0.23834013938903809, 0.2475297451019287, 0.23901677131652832, 0.24003887176513672, 0.2425374984741211, 0.2402210235595703, 0.24400544166564941, 0.23168635368347168, 0.24262380599975586, 0.2350749969482422, 0.2582244873046875, 0.24034643173217773, 0.2339801788330078, 0.24012494087219238, 0.24579238891601562, 0.23562026023864746, 0.2396249771118164, 0.24032998085021973, 0.2414085865020752, 0.2439882755279541, 0.23009705543518066, 0.24074816703796387, 0.23401188850402832, 0.23701119422912598, 0.23403620719909668, 0.2412559986114502, 0.2453327178955078, 0.24075627326965332, 0.24298548698425293, 0.23409199714660645, 0.23627662658691406, 0.23524808883666992, 0.24144768714904785, 0.25594115257263184, 0.23947787284851074, 0.2396399974822998, 0.23328733444213867, 0.23700380325317383, 0.24587273597717285, 0.2377610206604004, 0.23802733421325684, 0.23622727394104004, 0.23874402046203613, 0.24677181243896484, 0.24212908744812012, 0.26503992080688477, 0.23714876174926758, 0.2447967529296875, 0.24988532066345215, 0.2839953899383545, 0.23778986930847168, 0.2795255184173584, 0.6213498115539551, 0.3251307010650635, 0.2483363151550293, 0.2914128303527832, 0.2699720859527588, 0.26418066024780273, 0.28227806091308594, 0.26906514167785645, 0.24215245246887207, 0.23160219192504883, 0.24245667457580566, 0.2393956184387207, 0.23816585540771484, 0.24792098999023438, 0.2527143955230713, 0.2452235221862793, 0.2635645866394043, 0.2516026496887207, 0.23379278182983398, 0.26272010803222656, 0.2565474510192871, 0.24600839614868164, 0.2481985092163086, 0.2386312484741211, 0.24302458763122559, 0.2605321407318115, 0.2445080280303955, 0.24600625038146973, 0.23324799537658691, 0.2534170150756836, 0.23881196975708008, 0.23710370063781738, 0.2761225700378418, 0.23386454582214355, 0.24981427192687988, 0.24338650703430176, 0.2366337776184082, 0.24183130264282227, 0.2572784423828125, 0.25801563262939453, 0.2575411796569824, 0.24648761749267578, 0.2345895767211914, 0.253817081451416, 0.2641286849975586, 0.25049495697021484, 0.23800969123840332, 0.23590302467346191, 0.24261164665222168, 0.23650074005126953, 0.24204349517822266, 0.2360384464263916, 0.24192523956298828, 0.23588180541992188, 0.24002766609191895, 0.2548182010650635, 0.23165369033813477, 0.2551918029785156, 0.23196101188659668, 0.2439887523651123, 0.25673699378967285, 0.24646663665771484, 0.24794220924377441, 0.2440629005432129, 0.23309779167175293, 0.24945306777954102, 0.2739982604980469, 0.2705268859863281, 0.27605676651000977, 0.23929095268249512, 0.24223971366882324, 0.24358606338500977, 0.23523950576782227, 0.24786734580993652, 0.23691654205322266, 0.23952579498291016, 0.23717355728149414, 0.24467682838439941, 0.24887442588806152, 0.23400330543518066, 0.24906039237976074, 0.23716402053833008, 0.24747681617736816, 0.23867464065551758, 0.23693537712097168, 0.24185872077941895, 0.2460155487060547, 0.2331523895263672, 0.24695277214050293, 0.23880243301391602, 0.24179673194885254, 0.24566006660461426, 0.2321016788482666, 0.24119257926940918, 0.24083280563354492, 0.25397491455078125, 0.2612011432647705, 0.23568367958068848, 0.24121594429016113, 0.24387264251708984, 0.2349836826324463, 0.23909544944763184, 0.24598455429077148, 0.23862004280090332, 0.24904346466064453, 0.2392559051513672, 0.24093127250671387, 0.2438502311706543, 0.2334003448486328, 0.23999309539794922, 0.23693037033081055, 0.23758935928344727, 0.23600077629089355, 0.2387995719909668, 0.24256658554077148, 0.23592734336853027, 0.24489164352416992, 0.23780131340026855, 0.2423865795135498, 0.2441091537475586, 0.2412407398223877, 0.24712085723876953, 0.2329120635986328, 0.23828577995300293, 0.23051047325134277, 0.24341368675231934, 0.24897027015686035, 0.2393651008605957, 0.23919916152954102, 0.2319622039794922, 0.2420964241027832, 0.24513840675354004, 0.23450756072998047, 0.24370646476745605, 0.24044132232666016, 0.23737716674804688, 0.24786686897277832, 0.2366480827331543, 0.24584102630615234, 0.23693132400512695, 0.24042248725891113, 0.23139047622680664, 0.24115657806396484, 0.24378252029418945, 0.2329082489013672, 0.24376916885375977, 0.23280906677246094, 0.24036288261413574, 0.2360999584197998, 0.23877573013305664, 0.23223114013671875, 0.23699712753295898, 0.24686384201049805, 0.2413630485534668, 0.24350547790527344, 0.23714160919189453, 0.23549795150756836, 0.23571133613586426, 0.2388465404510498, 0.24327778816223145, 0.24208879470825195, 0.24133634567260742, 0.23324322700500488, 0.23946833610534668, 0.2439737319946289, 0.24230504035949707, 0.24212241172790527, 0.23401403427124023, 0.23989272117614746, 0.23314785957336426, 0.24187517166137695, 0.241041898727417, 0.23545336723327637]
    data = []
    file = open('./saved_models/PO3_v5/training_49.pckl', 'rb')
    # dump information to that file
    epoch_data = pickle.load(file)
    # close the file
    file.close()

    data += epoch_data[0][0][1:]
    data += epoch_data[1][0][1:]



    adjusted_data = [round(i,3) for i in data]
    print(adjusted_data)
    # adjusted_data = [round(round(i*5, 2)/5,3) for i in adjusted_data]
    print(adjusted_data)
    data_dict = dict()
    print(max(adjusted_data))
    uitschieters = 0
    for i in adjusted_data:
        if i > 0.35:
            uitschieters += 1
        else:
            data_dict[i] = data_dict.get(i, 0) + 1
    print(uitschieters, len(data), uitschieters/len(data))
    x = sorted(list(data_dict.keys()))
    y = [data_dict[i] for i in x]
    # Fixing random state for reproducibility


    print("median", statistics.median(data))
    print("mean", statistics.mean(data))
    print()

    fig1, ax1 = plt.subplots()
    ax1.set_title('Boxplot evaluation time')
    ax1.boxplot(data, vert=False, showfliers=False)
    fig1.show()


    plt.bar(range(len(y)), y, align="center")
    # plt.xticks(range(len(data_dict)), x, rotation='vertical')
    plt.xticks(range(0, len(x), 10), x[::10], rotation='vertical')
    plt.title("Barplot evalutaion time")
    plt.show()


if __name__ == "__main__":
    pass
    # choose function to plot

