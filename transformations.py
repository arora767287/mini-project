import cv2
import numpy as np
import imutils
from imutils import contours
import math
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import scipy, pylab


def matrix_multiplucation(array_one, array_two):
    length_one = len(array_one)
    #What if the input arrays do not have the same columnand row lengths respectively?
    length_two = len(array_two)
    new_array = []
    #array_two is always a set of coordinates
    if len(array_one[0]) == 2:
        for i in range(0,len(array_one)):
            new_x = array_one[i][0]*array_two[0][0] + array_one[i][1]*array_two[0][1]    
            new_y = array_one[i][0]*array_two[1][0] + array_one[i][1]*array_two[1][1]
            new_array.append([new_x, new_y])
    elif((len(array_one[0])==3) and (len(array_two[0])==3)):
        for i in range(0, len(array_one)):
            new_x = array_one[i][0]*array_two[0][0] + array_one[i][1]*array_two[0][1] + array_one[i][2]*array_two[0][2]
            new_y = array_one[i][0]*array_two[1][0] + array_one[i][1]*array_two[1][1] + array_one[i][2]*array_two[1][2]
            new_z = array_one[i][0]*array_two[2][0] + array_one[i][1]*array_two[2][1] + array_one[i][2]*array_two[2][2]
            new_array.append([new_x,new_y,new_z])
    return new_array
    

transformations = ["Rotation", "Translation", "Jump", "Reflection", "Dilation", "Shear"]
new_img = "/Users/Home/Downloads/unnamed-2.jpg"
newer_img = cv2.imread(new_img)
newer_img = np.array(newer_img)
gray_image = cv2.cvtColor(newer_img, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray_image, (5, 5), 1)
thresh = cv2.threshold(gray, 210, 255, cv2.THRESH_BINARY)[1]
thresh = cv2.erode(thresh, None, iterations=2)
thresh = cv2.dilate(thresh, None, iterations=2)
# find contours in thresholded image, then grab the largest
# one
thresh_new = cv2.bitwise_not(thresh.copy())
cnts = cv2.findContours(thresh_new.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
c = max(cnts, key=cv2.contourArea)
cv2.drawContours(newer_img, cnts, -1, (0, 255, 255), 1)
#Coordinate of biggest contour are stored here
print c
coordinates = []
for i in range(0, len(c)):
    for l in range(0,len(c[i])):
        coordinates.append([c[i][l][0], c[i][l][1]])
x_coordinates = []
y_coordinates = []
for i in range(0, len(coordinates)):
    x_coordinates.append(coordinates[i][0])

def rotate_special_x(original_array, input_angle):
    x =[]
    y=[]
    input_angle = input_angle*(math.pi)/180
    final_array = []
    for i in range(0, 80):
        one_angle = float(i+1)/80
        final_array.append(rotation(one_angle, original_array))
    for i in range(0,len(final_array)):
        x.append([])
        y.append([])
        for c in range(0,len(final_array[i])):
            x[i].append(final_array[i][c][0])
            y[i].append(final_array[i][c][0])
    return x,y
'''
def rotate_special_y(original_array, input_angle):
    y =[]
    input_angle = input_angle*(float(math.pi()))/180
    for i in range(0, len(original_array)):
        y.append(original_array[i][1])
    new_y = []
    for i in range(0, 80):
        one_angle = float(i+1)/80
        new_y.append(rotation(one_angle, y))
    return new_y
'''
def rotate_animate(original_array, input_angle):
    figure = plt.figure()
    ax1 = pylab.subplot(111)
    x_array = rotate_special_x(original_array, input_angle)[0]
    y_array = rotate_special_x(original_array, input_angle)[1]

    def animate1(i):
        ax1.scatter(x_array[i], y_array[i])
    ani = FuncAnimation(figure, animate1, interval=300, frames=80)
    plt.show()


def display_transformation(original_array, final_array):
    x=[]
    y=[]
    for i in range(0,len(original_array)):
        x.append(original_array[i][0])
        y.append(-1*original_array[i][1])
    x_coordinates_new = []
    y_coordinates_new = []
    for i in range(0, len(final_array)):
        x_coordinates_new.append(final_array[i][0])
        y_coordinates_new.append(-1*final_array[i][1])
    figure = plt.figure()
    axl = pylab.subplot(111)
    axl.scatter(x,y, s=10, c='b', marker="s", label='first')
    axl.scatter(x_coordinates_new,y_coordinates_new, s=10, c='r', marker="o", label='second')
    '''
    plt.scatter(x_coordinates, y_coordinates, linestyle="solid", color = 'blue')
    plt.plot(x_coordinates, y_coordinates)
    plt.title("Transformations")
    plt.xlabel("Coordinates")
    plt.ylabel("Coordinates")
    '''
    plt.show()

def rotation(angle, object_coordinates):
    angle_one = angle*(math.pi)/180
    rotation_two_matrix = [[math.cos(angle_one),-1*math.sin(angle_one)],[math.sin(angle_one),math.cos(angle_one)]]
    return matrix_multiplucation(object_coordinates, rotation_two_matrix)
def reflection_across_line(line_tuple, object_coordinates):
    a = line_tuple[0]
    b = line_tuple[1]
    c = line_tuple[2]
    z = ((a**2)+(b**2))
    x_m = float((b**2)-(a**2))/z
    x_n = float((-2*a*b))/z
    x_blank = float((-2*a*c))/z
    y_m = float(-2*a*b)/z
    y_n = float(((a**2)-(b**2)))/z
    y_blank = float((-2*b*c))/z
    reflection_across_line_matrix = [[x_m, x_n, x_blank],[y_m,y_n, y_blank],[0,0,1]]
    return matrix_multiplucation(object_coordinates, reflection_across_line_matrix)
def translation_three(translation_tuple, object_coordinates):
    horizontal_translation = translation_tuple[0]
    vertical_translation = translation_tuple[1]
    translation_three_matrix = [[1,0,horizontal_translation],[0,1,vertical_translation],[0,0,1]]
    return matrix_multiplucation(object_coordinates, translation_three_matrix)
def strech_two_x_matrix(multiplier_tuple, object_coordinates):
    x_multiplier = multiplier_tuple[0]
    y_multiplier = multiplier_tuple[1]
    shear_two_x_matrix = [[1,x_multiplier],[y_multiplier,1]]
    return matrix_multiplucation(object_coordinates, strech_two_matrix)
def dilation_two_matrix(dilation_multiplier, object_coordinates):
    x_multiplier = dilation_multiplier[0]
    y_multiplier = dilation_multiplier[1]
    dilation_two_matrix = [[x_multiplier, 0],[0, y_multiplier]]
    return matrix_multiplucation(object_coordinates, dilation_two_matrix)


transformations = ["Rotation", "Translation", "Reflection", "Dilation", "Shear", "Rotation Animation"]
input_image= input("What is the file path of the image you would like?")
input_t = input("What transformation would you like? Enter 1 for rotation, 2 for translation, 3 for dilation, and 4 for rotation animation")
if input_t == 1:
    input_angle = input("What angle would you like to rotate by?")
    display_transformation(coordinates, rotation(input_angle, coordinates))
elif input_t == 2:
    for i in range(0,len(coordinates)):
        coordinates[i].append(1)
    horizontal_translation = input("What would you like to translate horizontally by?")
    vertical_translation = input("What would you like to translate vertically by?")
    translation_tuple = [horizontal_translation, vertical_translation]
    display_transformation(coordinates,translation_three(translation_tuple, coordinates))
elif input_t == 4:
    input_angle = input("What angle would you like to animate through?")
    print rotation(45, coordinates)[79][1]
    print float(input_angle)/80
    rotate_animate(coordinates, input_angle)
if input_t == 3:
    x_mu = input("Please enter the x_multiplier: ")
    y_mu = input("Please enter the y_multiplier: ")
    dilation_multiplier_1 = [x_mu, y_mu]
    display_transformation(coordinates, dilation_two_matrix(dilation_multiplier_1,coordinates))


#For reference, in mathematical calculations, the given line of reflection is ax+by+c = 0
#new_coordinates = rotation(180, coordinates)
#display_transformation(coordinates, new_coordinates)

cv2.imshow('New Image', thresh)
cv2.imshow('Image',newer_img)
cv2.waitKey(100000)

