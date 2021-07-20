import cv2 as cv
import numpy as np

cap = cv.VideoCapture('/home/user/Downloads/Road Lanes.avi')


def gauss(image):
    return cv.GaussianBlur(image, (5, 5), 0)


def region(image):
    height, width = image.shape
    # isolate the gradients that correspond to the lane lines
    triangle = np.array([
        [(100, height), (475, 325), (width, height)]
    ])
    # create a black image with the same dimensions as original image
    mask = np.zeros_like(image)
    # create a mask (triangle that isolates the region of interest in our image)
    mask = cv.fillPoly(mask, triangle, 255)
    mask = cv.bitwise_and(image, mask)
    return mask


def display_lines(image, lines):
    lines_image = np.zeros_like(image)
    # make sure array isn't empty
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line
            # draw lines on a black image
            cv.line(lines_image, (x1, y1), (x2, y2), (255, 0, 0), 10)
    return lines_image


def average(image, lines):
    left = []
    right = []

    if lines is not None:
        for line in lines:
            print(line)
            x1, y1, x2, y2 = line.reshape(4)
            # fit line to points, return slope and y-int
            parameters = np.polyfit((x1, x2), (y1, y2), 1)
            print(parameters)
            slope = parameters[0]
            y_int = parameters[1]
            # lines on the right have positive slope, and lines on the left have neg slope
            if slope < 0:
                left.append((slope, y_int))
            else:
                right.append((slope, y_int))


def make_points(image, average):
    print(average)
    slope, y_int = average
    y1 = image.shape[0]
    # how long we want our lines to be --> 3/5 the size of the image
    y2 = int(y1 * (3 / 5))
    # determine algebraically
    x1 = int((y1 - y_int) // slope)
    x2 = int((y2 - y_int) // slope)
    return np.array([x1, y1, x2, y2])


while cap.isOpened:
    ret, frame = cap.read()
    if ret == True:
        # ----THE PREVIOUS ALGORITHM----#
        gaus = gauss(frame)
        edges = cv.Canny(gaus, 50, 150)
        isolated = region(edges)
        lines = cv.HoughLinesP(isolated, 2, np.pi / 180, 50, np.array([]), minLineLength=40, maxLineGap=5)
        averaged_lines = average(frame, lines)
        black_lines = display_lines(frame, averaged_lines)
        lanes = cv.addWeighted(frame, 0.8, black_lines, 1, 1)
        cv.imshow('frame', lanes)
        # ----THE PREVIOUS ALGORITHM----#

        if cv.waitKey(10) & 0xFF == ord('q'):
            break
    else:
        break

cap.release()
cv.destroyAllWindows()
