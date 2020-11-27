import tkinter as tk
import cv2
from PIL import Image, ImageTk
import time
import os
import sys
import numpy as np
from matplotlib import pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import math

# Social distance
social_distance = 1.5


# Turning on cameras and time
cam = cv2.VideoCapture(0)
#cam_2 = cv2.VideoCapture(2)
start_time = time.time()


# Resolutions
def make_1080p():
    cam.set(3, 1920)
    cam.set(4, 1080)
#    cam_2.set(3, 1920)
#    cam_2.set(4, 1080)

def make_720p():
    cam.set(3, 1280)
    cam.set(4, 720)
    cam_2.set(3, 1280)
    cam_2.set(4, 720)

def make_480p():
    cam.set(3, 640)
    cam.set(4, 480)
    cam_2.set(3, 640)
    cam_2.set(4, 480)

def change_res(width, height):
    cam.set(3, width)
    cam.set(4, height)
    cam_2.set(3, width)
    cam_2.set(4, height)

def rescale_frame(frame, percent=75):
    width = int(frame.shape[1] * percent/ 100)
    height = int(frame.shape[0] * percent/ 100)
    dim = (width, height)
    return cv2.resize(frame, dim, interpolation =cv2.INTER_AREA)

make_1080p()


# Turning on GUI
gui = tk.Tk()
#gui.configure(bg='blue')
gui.bind('<Escape>', lambda e: gui.quit())
gui.title('Stereo camera')
gui.resizable(True, True)


# Labels
label_L_camera = tk.Label(text= 'L-Camera')
label_L_camera.grid(row=0, column=0)
label_R_camera = tk.Label(text= 'R-Camera')
label_R_camera.grid(row=0, column=1)

label_distance = tk.Label(text = 'Closest distance:')
label_distance.grid(row=3, column=2, sticky=tk.N)
label_distance_show = tk.Label()
label_distance_show.grid(row=3, column=3, sticky=tk.N)

label_number_of_people = tk.Label(text= 'Number of people:')
label_number_of_people.grid(row=4, column=2, sticky=tk.N)
label_number_of_people_show = tk.Label()
label_number_of_people_show.grid(row=4, column=3, sticky=tk.N)

label_fps_L = tk.Label(text= 'FPS L:')
label_fps_L.grid(row=5, column=2, sticky=tk.N)
label_fps_R = tk.Label(text='FPS R:')
label_fps_R.grid(row=6, column=2, sticky=tk.N)
label_fps_1_show = tk.Label()
label_fps_1_show.grid(row=5, column=3, sticky=tk.N)
label_fps_2_show = tk.Label()
label_fps_2_show.grid(row=6, column=3, sticky=tk.N)

label_time = tk.Label(text = 'Time:')
label_time.grid(row=7, column=2, sticky=tk.N)
label_time_show = tk.Label()
label_time_show.grid(row=7, column=3, sticky=tk.N)

label_new_coordinate1 = tk.Label(text = 'New coordinate1: ')
label_new_coordinate1.grid(row=8, column=2, sticky=tk.N)
label_new_coordinate1_show = tk.Label()
label_new_coordinate1_show.grid(row=8, column=3, sticky=tk.N)

label_new_coordinate2 = tk.Label(text = 'New coordinate2: ')
label_new_coordinate2.grid(row=8, column=4, sticky=tk.N)
label_new_coordinate2_show = tk.Label()
label_new_coordinate2_show.grid(row=8, column=5, sticky=tk.N)

label_new_coordinate3 = tk.Label(text = 'New coordinate3: ')
label_new_coordinate3.grid(row=8, column=6, sticky=tk.N)
label_new_coordinate3_show = tk.Label()
label_new_coordinate3_show.grid(row=8, column=7, sticky=tk.N)

label_3D_plot = tk.Label(text = '3D Plot')
label_3D_plot.grid(row=2, column=0)


# Functions for buttons
def submit1():
    coordinate = make_tuple(entry_new_coordinates1.get())
    label_new_coordinate1_show.config(text=str(coordinate))
    if 'blue' in coordinate_dictionary:
        coordinate_dictionary['blue'] += [coordinate]
    else:
        coordinate_dictionary['blue'] = [coordinate]
#    global list_of_points1
#    list_of_points1 += [coordinate]
#    show_coordinates()

def submit2():
    coordinate = make_tuple(entry_new_coordinates2.get())
    label_new_coordinate2_show.config(text=str(coordinate))
    if 'white' in coordinate_dictionary:
        coordinate_dictionary['white'] += [coordinate]
    else:
        coordinate_dictionary['white'] = [coordinate]
#    global list_of_points2
#    list_of_points2 += [coordinate]
#    show_coordinates()

def submit3():
    coordinate = make_tuple(entry_new_coordinates3.get())
    label_new_coordinate3_show.config(text=str(coordinate))
    if 'green' in coordinate_dictionary:
        coordinate_dictionary['green'] += [coordinate]
    else:
        coordinate_dictionary['green'] = [coordinate]
#    global list_of_points3
#    list_of_points3 += [coordinate]
#    show_coordinates()

def reset():
    os.execl(sys.executable, os.path.abspath(__file__), *sys.argv)


# Manual coordinates
entry_new_coordinates1 = tk.Entry()
entry_new_coordinates1.grid(row=9, column=2, sticky=tk.N)
entry_new_coordinates2 = tk.Entry()
entry_new_coordinates2.grid(row=9, column= 4, sticky=tk.N)
entry_new_coordinates3 = tk.Entry()
entry_new_coordinates3.grid(row=9, column= 6, sticky=tk.N)

def make_tuple(string):
    begin = 1
    result = list()
    n = 1
    end = len(string) - 1
    while n != end:
        if string[n] == ',':
            result += [float(string[begin:n])]
            n = n + 1
            begin = n
        elif n == end-1:
            n = n + 1
            result += [float(string[begin:n])]
        else:
            n = n+1
    return tuple(result)

button_submit1 = tk.Button(text='Submit1', command=submit1)
button_submit1.grid(row=10, column=2, sticky=tk.N)
button_submit2 = tk.Button(text='Submit2', command=submit2)
button_submit2.grid(row=10, column=4, sticky=tk.N)
button_submit3 = tk.Button(text='Submit3', command=submit3)
button_submit3.grid(row=10, column=6, sticky=tk.N)



# Buttons
button_reset = tk.Button(text= 'Reset', command=reset, bg= 'red')
button_reset.grid(row=11, column=2, sticky=tk.N)


# Cameras
camera_display1 = tk.Label()
camera_display1.grid(row=1, column=0)
camera_display2 = tk.Label()
camera_display2.grid(row=1, column=1)


# Functions
def show_frame1():
    _, frame1 = cam.read()
    frame1 = rescale_frame(frame1, 50)

    # Rode lijnen tekenen
    for element in ToClose:
        # element = (("Naam1",([3D-positie],[2D punt op linkse foto], [2D punt op rechtse foto], [bounding box op linkse foto], [bounding box op rechtse foto])),
        #           ("Naam2", ([3D-positie],[2D punt op linkse foto], [2D punt op rechtse foto], [bounding box op linkse foto], [bounding box op rechtse foto])))
        point1 = element[0][1][1]
        point2 = element[1][1][1]
        cv2.line(frame1,
                 point1,  # coördinaat 1
                 point2,  # coördinaat 2
                 (0, 0, 255),  # kleur rood
                 5)  # dikte lijn

    # Bounding Boxes tekenen
#    ListData = dict.values()
#    for element in ListData:
#        box = element[4]
#        cv2.rectangle(frame1, (box[0], box[1]), (box[2], box[3]), (50, 50, 200))

    # Gekopieerd van gui4. Weet niet of dit nog steeds gaat werken. Ik gebruikte andere functies om frame te displayen
    cv2image1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGBA)
    img1 = Image.fromarray(cv2image1)
    imgtk1 = ImageTk.PhotoImage(image=img1)
    camera_display1.imgtk1 = imgtk1
    camera_display1.configure(image=imgtk1)
    camera_display1.after(10, show_frame1)

def show_frame2():
    _, frame2 = cam_2.read()
    frame2 = rescale_frame(frame2, 50)

    # Rode lijnen tekenen
    for element in ToClose:
        # element = (("Naam1",([3D-positie],[2D punt op linkse foto], [2D punt op rechtse foto], [bounding box op linkse foto], [bounding box op rechtse foto])),
        #           ("Naam2", ([3D-positie],[2D punt op linkse foto], [2D punt op rechtse foto], [bounding box op linkse foto], [bounding box op rechtse foto])))
        point1 = element[0][1][2]
        point2 = element[1][1][2]
        cv2.line(frame1,
                 point1,  # coördinaat 1
                 point2,  # coördinaat 2
                 (0, 0, 255),  # kleur rood
                 5)  # dikte lijn

    # Bounding Boxes tekenen
#    ListData = dict.values()
#    for element in ListData:
#        box = element[5]
#        cv2.rectangle(frame2, (box[0], box[1]), (box[2], box[3]), (50, 50, 200))

    # Gekopieerd van gui4. Weet niet of dit nog steeds gaat werken. Ik gebruikte andere functies om frame te displayen
    cv2image2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGBA)
    img2 = Image.fromarray(cv2image2)
    imgtk2 = ImageTk.PhotoImage(image=img2)
    camera_display2.imgtk2 = imgtk2
    camera_display2.configure(image=imgtk2)
    camera_display2.after(10, show_frame2)

def print_information():
    if label_distance_show.cget('fg') == 'red':
        start_danger_time = time.time()
        while label_distance_show.cget('fg') == 'red':
            ...
        stop_danger_time = time.time()
        danger = stop_danger_time - start_danger_time
        print('Person blue and person white were to close for:')
        print(str(danger) + ' seconds')
        print('This occured between: ' + str(start_danger_time) + ' and ' + str(stop_danger_time))
    label_distance_show.after(10, print_information)

def show_distance():
    closest_distance = None
    for person_1 in coordinate_dictionary:
        for person_2 in coordinate_dictionary:
            if person_1 != person_2:
                distance = calculate_distance(person_1, person_2)
                if closest_distance == None or distance < closest_distance:
                    closest_distance = distance
    label_distance_show.configure(text=str(closest_distance), fg= 'red' if (isinstance(closest_distance, float) and closest_distance < social_distance) else 'black')
    label_distance_show.after(10, show_distance)

def show_time():
    end_time = time.time()
    time_difference = str(round(end_time - start_time))
    label_time_show.configure(text= time_difference)
    label_time_show.after(10, show_time)

def show_number_of_people():
    number_of_people = get_number_of_people()
    label_number_of_people_show.configure(text= str(number_of_people))
    label_number_of_people_show.after(10, show_number_of_people)

def show_coordinates():
    number = 0
    for person in coordinate_dictionary:
        number += 1
        list_of_points = get_list_of_points(person)
        number_of_coordinates = len(list_of_points)
        q = give_q(list_of_points)
        x = get_x(q)
        y = get_y(q)
        z = get_z(q)

        if person not in stick_figure_dictionary or len(get_list_of_bodies(person)) < number_of_coordinates:
            points, = ax.plot(x, y, z, stick_figure_colors[number-1] + 'o')
            body, = ax.plot([x[-1], x[-1]], [y[-1], y[-1]], [z[-1]+0.9, z[-1] + 1.8], stick_figure_colors[number-1])
            left_leg, = ax.plot([x[-1] - 0.2, x[-1]], [y[-1], y[-1]], [z[-1], z[-1] + 0.9], stick_figure_colors[number-1])
            right_leg, = ax.plot([x[-1] + 0.2, x[-1]], [y[-1], y[-1]], [z[-1], z[-1] + 0.9], stick_figure_colors[number-1])
            left_arm, = ax.plot([x[-1] - 0.2, x[-1]], [y[-1], y[-1]], [z[-1] + 1, z[-1] + 1.6], stick_figure_colors[number-1])
            right_arm, = ax.plot([x[-1] + 0.2, x[-1]], [y[-1], y[-1]], [z[-1] + 1, z[-1] + 1.6], stick_figure_colors[number-1])
            heads, = ax.plot([x[-1]], [y[-1]], [z[-1] + 1.8], stick_figure_colors[number-1] + 'o', markersize=14)

            if person in stick_figure_dictionary:
                stick_figure_dictionary[person][0] += [body]
                stick_figure_dictionary[person][1] += [left_leg]
                stick_figure_dictionary[person][2] += [right_leg]
                stick_figure_dictionary[person][3] += [left_arm]
                stick_figure_dictionary[person][4] += [right_arm]
                stick_figure_dictionary[person][5] += [heads]
            else:
                stick_figure_dictionary[person] = [[body], [left_leg], [right_leg], [left_arm], [right_arm], [heads]]

            if len(list_of_points) > 1:
                lines, = ax.plot([x[-2], x[-1]], [y[-2], y[-1]], [z[-2], z[-1]], stick_figure_colors[number-1])
                get_list_of_bodies(person)[-2].remove()
                get_list_of_left_legs(person)[-2].remove()
                get_list_of_right_legs(person)[-2].remove()
                get_list_of_left_arms(person)[-2].remove()
                get_list_of_right_arms(person)[-2].remove()
                get_list_of_heads(person)[-2].remove()

            if get_number_of_people() > 1:
                for person_1 in coordinate_dictionary:
                    for person_2 in coordinate_dictionary:
                        if person_1 != person_2:
                            distance = calculate_distance(person_1, person_2)
                            if distance < social_distance:
                                list_of_points_1 = get_list_of_points(person_1)
                                list_of_points_2 = get_list_of_points(person_2)
                                x1 = list_of_points_1[-1][0]
                                x2 = list_of_points_2[-1][0]
                                y1 = list_of_points_1[-1][1]
                                y2 = list_of_points_2[-1][1]
                                z1 = list_of_points_1[-1][2]
                                z2 = list_of_points_2[-1][2]
                                redline, = ax.plot([x1, x2], [y1, y2], [z1, z2], 'r')

    plt.draw()
#    plt.pause(0.02)
    gui.after(500, show_coordinates)

def show_fps1():
    fps1 = cam.get(cv2.CAP_PROP_FPS)
    label_fps_1_show.configure(text=str(fps1))
    label_fps_1_show.after(10, show_fps1)

def show_fps2():
    fps2 = cam_2.get(cv2.CAP_PROP_FPS)
    label_fps_1_show.configure(text=str(fps2))
    label_fps_1_show.after(10, show_fps2)

# 3D PLOT
def numpy_to_list(list_of_points):
    k=0
    x_list = []
    y_list = []
    z_list = []
    while k < len(list_of_points):
        x_list.append(list_of_points[k][0])
        y_list.append(list_of_points[k][1])
        z_list.append(list_of_points[k][2])
        k+=1
    return (x_list,y_list,z_list)


#definiëren coördinaten
coordinate_dictionary = dict()
stick_figure_dictionary = dict()
stick_figure_colors = ['b', 'w', 'g', 'y', 'k', 'm', 'c']

def calculate_distance(name1, name2):
    list_of_points1 = get_list_of_points(name1)
    list_of_points2 = get_list_of_points(name2)
    x1 = list_of_points1[-1][0]
    x2 = list_of_points2[-1][0]
    y1 = list_of_points1[-1][1]
    y2 = list_of_points2[-1][1]
    z1 = list_of_points1[-1][2]
    z2 = list_of_points2[-1][2]
    distance = math.sqrt((x2-x1)**2 + (y2-y1)**2 + (z2-z1)**2)
    return distance

def get_to_close_distances(dict):
    ListNames = dict.keys()
    ListData = dict.values()
    ToClose = []
    k = 0
    v = 1
    while k < len(ListData) - 1:
        while v < len(ListData):
            x1, y1 = ListData[k][0], ListData[k][1]
            x2, y2 = ListData[v][0], ListData[v][0]
            distance = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** (1 / 2)
            if distance < 1.5:
                ToClose.append(((ListNames[k], ListData[k]), (ListNames[v], ListData[v])))
            v += 1
        k += 1
        v = k + 1
    return ToClose
ToClose = get_to_close_distances(dict())

def get_number_of_people():
    return len(coordinate_dictionary)

def get_list_of_points(name):
    return coordinate_dictionary[name]

def get_list_of_bodies(name):
    return stick_figure_dictionary[name][0]

def get_list_of_left_legs(name):
    return stick_figure_dictionary[name][1]

def get_list_of_right_legs(name):
    return stick_figure_dictionary[name][2]

def get_list_of_left_arms(name):
    return stick_figure_dictionary[name][3]

def get_list_of_right_arms(name):
    return stick_figure_dictionary[name][4]

def get_list_of_heads(name):
    return stick_figure_dictionary[name][5]

def give_q(list_of_pts):
    x_list = numpy_to_list(list_of_pts)[0]
    y_list = numpy_to_list(list_of_pts)[1]
    z_list = numpy_to_list(list_of_pts)[2]
    q = [x_list, y_list, z_list]
    return q


#instellen plot
fig = plt.figure()

# Canvas
canvas = FigureCanvasTkAgg(fig, master=gui)  # A tk.DrawingArea.
canvas.draw()
canvas.get_tk_widget().grid(row=3, column=0, rowspan=20)

# Axis
ax = p3.Axes3D(fig)

def get_x(q_test):
    x = np.array(q_test[0])
    return x

def get_y(q_test):
    y = np.array(q_test[1])
    return y

def get_z(q_test):
    z = np.array(q_test[2])
    return z

txt = fig.suptitle('Virtual Room')

xmin = 0
xmax = 10
ymin = 0
ymax = 10
zmin = 0
zmax = 3

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')

axes = plt.gca()
axes.set_xlim([xmin, xmax])
axes.set_ylim([ymin, ymax])
axes.set_zlim([zmin, zmax])

fig.set_facecolor('gray')
ax.set_facecolor('gray')
ax.w_xaxis.pane.fill = False
ax.w_yaxis.pane.fill = False
ax.w_zaxis.pane.fill = False

# Show
show_frame1()
#show_frame2()

show_fps1()
#show_fps2()

show_distance()

show_time()

show_number_of_people()

show_coordinates()

#print_information()


gui.mainloop()