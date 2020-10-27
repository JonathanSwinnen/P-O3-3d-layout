import tkinter as tk
import cv2
from PIL import Image, ImageTk
import time
import os
import sys
import numpy as np
from matplotlib import pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk


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


gui = tk.Tk()
gui.bind('<Escape>', lambda e: gui.quit())
gui.title('Stereo camera')
gui.resizable(True, True)


# Labels
label_L_camera = tk.Label(text= 'L-Camera')
label_L_camera.grid(row=0, column=0)
label_R_camera = tk.Label(text= 'R-Camera')
label_R_camera.grid(row=0, column=1)

label_distance = tk.Label(text = 'Closest distance:')
label_distance.grid(row=5, column=2)

label_number_of_people = tk.Label(text= 'Number of people:')
label_number_of_people.grid(row=6, column=2)

label_fps_L = tk.Label(text= 'FPS L:')
label_fps_L.grid(row=7, column=2)
label_fps_R = tk.Label(text='FPS R:')
label_fps_R.grid(row=8, column=2)
label_fps_1_show = tk.Label()
label_fps_1_show.grid(row=7, column=3)
label_fps_2_show = tk.Label()
label_fps_2_show.grid(row=8, column=3)

label_time = tk.Label(text = 'Time:')
label_time.grid(row=9, column=2)
label_time_show = tk.Label()
label_time_show.grid(row=9, column=3)

label_3D_plot = tk.Label(text = '3D Plot')
label_3D_plot.grid(row=2, column=0)
label_toolbar = tk.Label()
label_toolbar.grid(row=4, column=0)

# Buttons
def reset():
    os.execl(sys.executable, os.path.abspath(__file__), *sys.argv)

button_reset = tk.Button(text= 'Reset', command=reset, bg= 'red')
button_reset.grid(row=10, column=2)

# Cameras
camera_display1 = tk.Label(gui)
camera_display1.grid(row=1, column=0)
camera_display2 = tk.Label(gui)
camera_display2.grid(row=1, column=1)


# Functions
def show_frame1():
    _, frame1 = cam.read()
    frame1 = rescale_frame(frame1, 50)
    frame1 = cv2.flip(frame1, 1)
    cv2image1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGBA)
    img1 = Image.fromarray(cv2image1)
    imgtk1 = ImageTk.PhotoImage(image=img1)
    camera_display1.imgtk1 = imgtk1
    camera_display1.configure(image=imgtk1)
    camera_display1.after(10, show_frame1)

def show_frame2():
    _, frame2 = cam_2.read()
    frame2 = rescale_frame(frame2, 50)
    frame2 = cv2.flip(frame2, 1)
    cv2image2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGBA)
    img2 = Image.fromarray(cv2image2)
    imgtk2 = ImageTk.PhotoImage(image=img2)
    camera_display2.imgtk2 = imgtk2
    camera_display2.configure(image=imgtk2)
    camera_display2.after(10, show_frame2)

def show_time():
    end_time = time.time()
    time_difference = str(round(end_time - start_time))
    label_time_show.configure(text= time_difference)
    label_time_show.after(10, show_time)

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

#afstand berekenen tussen alle punten
def distance_calculation(x,y,z):
    dist = []
    q=1
    k=0
    while k < len(x)-1:
        while q < len(x)-1:
            afstand = ((x[q] - x[k]) ** 2 + (y[q] - y[k]) ** 2 + (
                        z[q] - z[k]) ** 2) ** (1 / 2)
            if afstand < 1.5:
                dist.append((afstand,[x[q],x[k]],[y[q],y[k]],[z[q],z[k]]))
            q+=1
        k += 1
        q=k+1
    return dist

#definiëren coördinaten
list_of_points = [(1,5,1.78),(7,2,1.73),(4,1,1.55),(1,4,1.54),
                  (2,3,1.65),(7,3,1.94)] #moet van camera komen
x_list = numpy_to_list(list_of_points)[0]
y_list = numpy_to_list(list_of_points)[1]
z_list = numpy_to_list(list_of_points)[2]

q = [x_list,y_list,z_list]

#instellen plot
fig = plt.figure()

# Canvas
canvas = FigureCanvasTkAgg(fig, master=gui)  # A tk.DrawingArea.
canvas.draw()
canvas.get_tk_widget().grid(row=3, column=0)

# Axis
ax = p3.Axes3D(fig)

x=np.array(q[0])
y=np.array(q[1])
z=np.array(q[2])

points, = ax.plot(x, y, z, 'o')
txt = fig.suptitle('')

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

distances = distance_calculation(x,y,z)

#rode lijnen plotten
k=0
while k < len(distances):
    plt.plot(distances[k][1], distances[k][2], distances[k][3], 'r')
    k+=1

# Toolbar
toolbar = NavigationToolbar2Tk(canvas, label_toolbar)
toolbar.update()

# Show
show_frame1()
#show_frame2()

show_fps1()
#show_fps2()

show_time()


gui.mainloop()