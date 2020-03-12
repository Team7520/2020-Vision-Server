#!/usr/bin/env python3

'''Defines a class for which each year's subclass vision server inherits from'''

import sys
import time
import cv2
import numpy
import logging
import json
import math
from threading import Thread
from datetime import datetime

from math import sin, cos, atan2  # , degrees
from numpy import array, zeros, pi, sqrt, uint8

import cscore
# from wpilib import SmartDashboard, SendableChooser
from cscore.imagewriter import ImageWriter
from networktables.util import ntproperty, ChooserControl
from networktables import NetworkTables

two_pi = 2 * math.pi
pi_by_2 = math.pi / 2

#image size ratioed to 16:9
IMAGE_WIDTH = 424
IMAGE_HEIGHT = 240
CAMERA_FPS = 20
OUTPUT_FPS_LIMIT = 15

# Ball HSV setting
BALL_LOW_HSV = numpy.array((20, 100, 70), dtype=numpy.uint8)
BALL_HIGH_HSV = numpy.array((100, 255, 255), dtype=numpy.uint8)
BALL_DIAMETER = 7            # inches
KNOWN_DISTANCE = 48.0        # inches
KNOWN_RADIUS = 25.5        #

FLOOR_CAMERA_ANGLE = 0
SHOOTER_CAMERA_ANGLE = -7.5

# Goal HSV Setting
GOAL_LOW_HSV = numpy.array((55, 75, 55), dtype=numpy.uint8)
GOAL_HIGH_HSV = numpy.array((100, 255, 255), dtype=numpy.uint8)

#Lifecam 3000 from datasheet
#Datasheet: https://dl2jx7zfbtwvr.cloudfront.net/specsheets/WEBC1010.pdf
diagonalView = math.radians(68.5)

#16:9 aspect ratio
horizontalAspect = 16
verticalAspect = 9

#Reasons for using diagonal aspect is to calculate horizontal field of view.
diagonalAspect = math.hypot(horizontalAspect, verticalAspect)
#Calculations: http://vrguy.blogspot.com/2013/04/converting-diagonal-field-of-view-and.html
horizontalView = math.atan(math.tan(diagonalView/2) * (horizontalAspect / diagonalAspect)) * 2
verticalView = math.atan(math.tan(diagonalView/2) * (verticalAspect / diagonalAspect)) * 2

#Focal Length calculations: https://docs.google.com/presentation/d/1ediRsI-oR3-kwawFJZ34_ZTlQS2SDBLjZasjzZ-eXbQ/pub?start=false&loop=false&slide=id.g12c083cffa_0_165
H_FOCAL_LENGTH = IMAGE_WIDTH / (2*math.tan((horizontalView/2)))
V_FOCAL_LENGTH = IMAGE_HEIGHT / (2*math.tan((verticalView/2)))

#CALIB_STRING = '{"camera_matrix": [[374.32651400661666, 0.0, 208.1795122696603], [0.0, 374.0335899317395, 109.18785178761078], [0.0, 0.0, 1.0]], "distortion": [[0.11576562153880979, -0.8049683080723996, 0.0013190574515076887, 0.0014151850466233517, 1.1683306729610425]]}'
#CALIB_STRING = '{"camera_matrix": [[174.59631490154996, 0.0, 204.06067374445195], [0.0, 171.8900374073426, 49.797851023160234], [0.0, 0.0, 1.0]], "distortion": [[-0.0028047643695765546, 0.005030929501413801, -0.00418289030526314, -0.0026789800621453896, -0.00281275201996673]]}'  #20200229m
CALIB_STRING = '{"camera_matrix": [[428.6980048244047, 0.0, 214.40193947713368], [0.0, 425.7501289952879, 91.94700682430316], [0.0, 0.0, 1.0]], "distortion": [[0.6850803943638851, -6.057680289435895, 0.013676112211948982, -0.0007726317137721686, 15.244322530009654]]}'
C930E_CALIB_STRING = '{"camera_matrix": [[257.64544796248464, 0.0, 209.408398407944], [0.0, 258.1095404990428, 119.91831227387262], [0.0, 0.0, 1.0]], "distortion": [[0.07941072961742286, -0.20643561157601545, 6.806679344019089e-05, -0.00048416887625856066, 0.08253834888278783]]}'

# RMS: 0.1829526978177933
# camera matrix:
#  [[373.31641495   0.         210.90913851]
#  [  0.         373.0841807  113.30714513]
#  [  0.           0.           1.        ]]
# distortion coefficients:  [ 1.84199750e-01 -1.48673528e+00  2.11833158e-03  2.51929227e-04
#   2.87637488e+00]
# image center = (208.18, 109.19)
# FOV = (58.12, 35.85) degrees
# mtx = [[374.32651400661666, 0.0, 208.1795122696603], [0.0, 374.0335899317395, 109.18785178761078], [0.0, 0.0, 1.0]]
# dist = [[0.11576562153880979, -0.8049683080723996, 0.0013190574515076887, 0.0014151850466233517, 1.1683306729610425]]


def hough_fit(contour, nsides=None, approx_fit=None):
    '''Use the Hough line finding algorithm to find a polygon for contour.
    It is faster if you can provide an decent initial fit - see approxPolyDP_adaptive().'''

    if approx_fit is not None:
        nsides = len(approx_fit)
    if not nsides:
        raise Exception("You need to set nsides or pass approx_fit")

    x, y, w, h = cv2.boundingRect(contour)
    offset_vec = array((x, y))

    shifted_con = contour - offset_vec

    # the binning does affect the speed, so tune it....
    with CodeTimer("HoughLines"):
        contour_plot = zeros(shape=(h, w), dtype=uint8)
        cv2.drawContours(contour_plot, [shifted_con, ], -1, 255, 1)
        lines = cv2.HoughLines(contour_plot, 1, pi / 180, threshold=10)

    if lines is None or len(lines) < nsides:
        # print("HoughLines found too few lines")
        return None

    if approx_fit is not None:
        res = _match_lines_to_fit(approx_fit - offset_vec, lines, w, h)
    else:
        res = _find_sides(nsides, lines, w, h)

    if res is None:
        return None
    return array(res) + offset_vec


def approxPolyDP_adaptive(contour, nsides, max_dp_error=0.1):
    '''Use approxPolyDP to fit a polygon to a contour.
    Find the smallest dp_error that gets the correct number of sides.
    The results seem to often be a little wrong, but they are a quick starting point.'''

    step = 0.01
    peri = cv2.arcLength(contour, True)
    dp_err = step
    while dp_err <= max_dp_error:
        res = cv2.approxPolyDP(contour, dp_err * peri, True)
        if len(res) <= nsides:
            return res
        dp_err += step
    return None


def plot_hough_line(frame, rho, theta, color, thickness=1):
    '''Given (rho, theta) of a line in Hesse form, plot it on a frame.
    Useful for debugging, mostly.'''

    a = cos(theta)
    b = sin(theta)
    x0 = a * rho
    y0 = b * rho
    pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
    pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
    cv2.line(frame, pt1, pt2, color, thickness)
    return


# --------------------------------------------------------------------------------
# Private routines


def _find_sides(nsides, hough_lines, w, h):
    # The returned lines from HoughLines() are ordered by confidence, but there may/will be
    #  many variants of the best lines. Loop through the lines and pick the best from
    #  each cluster.

    contour_center = (w / 2, h / 2)
    boundaries = (-5, w+5, -5, h+5)

    dist_thres = 10
    theta_thres = pi / 36  # 5 degrees
    best_lines = []
    for linelist in hough_lines:
        line = linelist[0]
        if line[0] < 0:
            line[0] *= -1
            line[1] -= pi

        coord_near_ref = _compute_line_near_reference(line, contour_center)

        if not best_lines or not _is_close(best_lines, line, coord_near_ref, dist_thres, theta_thres):
            # print('best line:', line[0], math.degrees(line[1]))
            best_lines.append((line, coord_near_ref))

        if len(best_lines) == nsides:
            break

    if len(best_lines) != nsides:
        # print("hough_fit: found %s lines" % len(best_lines))
        return None

    # print('best')
    # for l in best_lines:
    #     print('   ', l[0][0], degrees(l[0][1]))

    # Find the nsides vertices which are inside the bounding box (with a little slop).
    # There will be extra intersections. Assume the right ones (and only those) are within the bounding box.
    vertices = []
    iline1 = 0
    used = set()
    used.add(iline1)
    while len(used) < nsides:
        found = False
        for iline2 in range(nsides):
            if iline2 in used:
                continue

            inter = _intersection(best_lines[iline1][0], best_lines[iline2][0])
            if inter is not None and \
               inter[0] >= boundaries[0] and inter[0] <= boundaries[1] and \
               inter[1] >= boundaries[2] and inter[1] <= boundaries[3]:
                vertices.append(inter)
                used.add(iline2)
                iline1 = iline2
                found = True
                break
        if not found:
            # print("No intersection with %s and available lines" % iline1)
            return None

    # add in the last pair
    inter = _intersection(best_lines[0][0], best_lines[iline1][0])
    if inter is not None and \
       inter[0] >= boundaries[0] and inter[0] <= boundaries[1] and \
       inter[1] >= boundaries[2] and inter[1] <= boundaries[3]:
        vertices.append(inter)

    if len(vertices) != nsides:
        # print('Not correct number of vertices:', len(vertices))
        return None

    # remember to unshift the resulting contour
    return vertices


def _delta_angle(a, b):
    d = a - b
    return (d + pi) % two_pi - pi


def _match_lines_to_fit(approx_fit, hough_lines, w, h):
    '''Given the approximate shape and a set of lines from the Hough algorithm
    find the matching lines and rebuild the fit'''

    theta_thres = pi / 36  # 5 degrees
    nsides = len(approx_fit)
    fit_sides = []
    hough_used = set()
    for ivrtx in range(nsides):
        ivrtx2 = (ivrtx + 1) % nsides
        pt1 = approx_fit[ivrtx][0]
        pt2 = approx_fit[ivrtx2][0]

        rho, theta = _hesse_form(pt1, pt2)

        # Hough lines are in order of confidence, so look for the first unused one
        #  which matches the line
        for ih, linelist in enumerate(hough_lines):
            if ih in hough_used:
                continue
            line = linelist[0]

            # There is an ambiguity of -rho and adding 180deg to theta
            # So test them both.

            if (abs(rho - line[0]) < 10 and abs(_delta_angle(theta, line[1])) < theta_thres) or \
               (abs(rho + line[0]) < 10 and abs(_delta_angle(theta, line[1] - pi)) < theta_thres):
                fit_sides.append(line)
                hough_used.add(ih)
                break

    if len(fit_sides) != nsides:
        # print("did not match enough lines")
        return None

    vertices = []
    for ivrtx in range(nsides):
        ivrtx2 = (ivrtx + 1) % nsides
        inter = _intersection(fit_sides[ivrtx], fit_sides[ivrtx2])
        if inter is None:
            # print("No intersection between lines")
            return None
        vertices.append(inter)

    return vertices


def _compute_line_near_reference(line, ref_point):
    with CodeTimer("compute_line_near_reference"):
        rho, theta = line

        # remember: theta is actually perpendicular to the line, so there is a sign flip
        cos_theta = cos(theta)
        sin_theta = sin(theta)
        x0 = cos_theta * rho
        y0 = sin_theta * rho

        if abs(cos_theta) < 1e-6:
            x_near_ref = None
            y_near_ref = y0
        elif abs(sin_theta) < 1e-6:
            x_near_ref = x0
            y_near_ref = None
        else:
            x_near_ref = x0 + (y0 - ref_point[1]) * sin_theta / cos_theta
            y_near_ref = y0 + (x0 - ref_point[0]) * cos_theta / sin_theta

    return x_near_ref, y_near_ref


def _is_close(best_lines, candidate, coord_near_ref, dist_thres, theta_thres):
    cand_rho, cand_theta = candidate

    # print('cand:', cand_rho, math.degrees(cand_theta))
    for line in best_lines:
        line, best_near_ref = line
        # print('best', line, best_near_ref)

        delta_dists = []
        if coord_near_ref[0] is not None and best_near_ref[0] is not None:
            delta_dists.append(abs(coord_near_ref[0] - best_near_ref[0]))
        if coord_near_ref[1] is not None and best_near_ref[1] is not None:
            delta_dists.append(abs(coord_near_ref[1] - best_near_ref[1]))
        if not delta_dists:
            return True
        delta_dist = min(delta_dists)

        # angle differences greater than 180deg are not real
        delta_theta = cand_theta - line[1]
        while delta_theta >= pi_by_2:
            delta_theta -= pi
        while delta_theta <= -pi_by_2:
            delta_theta += pi
        delta_theta = abs(delta_theta)

        # print('test:', line[0], math.degrees(line[1]), delta_dist, delta_theta)
        if delta_dist <= dist_thres and delta_theta <= theta_thres:
            return True
    return False


def _intersection(line1, line2):
    """Finds the intersection of two lines given in Hesse normal form.

    Returns closest integer pixel locations.
    See https://stackoverflow.com/a/383527/5087436
    """

    with CodeTimer("intersection"):
        rho1, theta1 = line1
        rho2, theta2 = line2
        if abs(theta1 - theta2) < 1e-6:
            # parallel
            return None

        cos1 = cos(theta1)
        sin1 = sin(theta1)
        cos2 = cos(theta2)
        sin2 = sin(theta2)

        denom = cos1*sin2 - sin1*cos2
        x = (sin2*rho1 - sin1*rho2) / denom
        y = (cos1*rho2 - cos2*rho1) / denom
        res = array((x, y))
    return res


def _hesse_form(pt1, pt2):
    '''Compute the Hesse form for the line through the points'''

    delta = pt2 - pt1
    mag2 = delta.dot(delta)
    vec = pt2 - pt2.dot(delta) * delta / mag2

    rho = sqrt(vec.dot(vec))
    if abs(rho) < 1e-6:
        # through 0. Need to compute theta differently
        theta = atan2(delta[1], delta[0]) + pi_by_2
        if theta > two_pi:
            theta -= two_pi
    else:
        theta = atan2(vec[1], vec[0])

    return rho, theta

class CodeTimer:
    timerDict = {}

    def __init__(self, name):
        self.name = name
        self.startT = None
        return

    def __enter__(self):
        self.startT = time.time()
        return

    def __exit__(self, exc_type, exc_value, traceback):
        dt = time.time() - self.startT
        entry = CodeTimer.timerDict.get(self.name, None)
        if entry is None:
            CodeTimer.timerDict[self.name] = [self.name, 1, dt]
        else:
            entry[1] += 1
            entry[2] += dt

        return

    @staticmethod
    def output_timers():
        for v in sorted(CodeTimer.timerDict.values(), key=lambda s: -s[2]):
            print("{0}: {1} frames in {2:.3f} sec: {3:.3f} ms/call, {4:.2f} calls/sec".format(v[0], v[1], v[2], 1000.0 * v[2]/float(v[1]), v[1]/v[2]))
        return

    @staticmethod
    def clear_timers():
        CodeTimer.timerDict.clear()
        return

class ThreadedCamera:
    '''Threaded camera reader. For now, a thin wrapper around the CSCore classes.'''

    def __init__(self, sink):
        '''Remember the source and set up the reading loop'''

        self.sink = sink
        self.timer = None

        self.frametime = None
        self.camera_frame = None

        # initialize the variable used to indicate if the thread should
        # be stopped
        self.stopped = False
        self.frame_number = 0
        self.last_read = 0
        return

    def start(self):
        '''Start the thread to read frames from the video stream'''

        t = Thread(target=self.update, args=())
        t.daemon = True
        t.start()
        return self

    def update(self):
        '''Threaded read loop'''

        fps_startt = time.time()

        while True:
            # if the thread indicator variable is set, stop the thread
            if self.stopped:
                return

            # otherwise, read the next frame from the stream
            self.frametime, self.camera_frame = self.sink.grabFrame(self.camera_frame)
            self.frame_number += 1

            if self.frame_number % 150 == 0:
                endt = time.time()
                dt = endt - fps_startt
                # logging.info("threadedcamera: 150 frames in {0:.3f} seconds = {1:.2f} FPS".format(dt, 150.0 / dt))
                fps_startt = endt
        return

    def next_frame(self):
        '''Wait for a new frame'''

        while self.last_read == self.frame_number:
            time.sleep(0.001)
        self.last_read = self.frame_number
        return self.frametime, self.camera_frame

    def get_frame(self):
        '''Return the frame most recently read, no waiting. This may be a repeat of the previous image.'''

        return self.frametime, self.camera_frame

    def stop(self):
        # indicate that the thread should be stopped
        self.stopped = True
        self.timer.output()
        time.sleep(0.3)         # time for thread to stop (proper way???)
        return

class Camera:
    '''Wrapper for camera related functionality.
    Makes handling different camera models easier
    Includes a threaded reader, so you can grab a frame without waiting, if needed'''

    def __init__(self, camera_server, name, device, height=240, fps=30, width=320, rotation=0,
                 threaded=False):
        '''Create a USB camera and configure it.
        Note: rotation is an angle: 0, 90, 180, -90'''

        self.width = int(width)
        self.height = int(height)
        self.rot90_count = (rotation // 90) % 4  # integer division

        self.camera = cscore.UsbCamera(name, device)
        camera_server.startAutomaticCapture(camera=self.camera)
        # keep the camera open for faster switching
        self.camera.setConnectionStrategy(cscore.VideoSource.ConnectionStrategy.kKeepOpen)

        self.camera.setResolution(self.width, self.height)
        self.camera.setFPS(int(fps))

        # set the camera for no auto focus, focus at infinity
        # NOTE: order does matter
        self.set_property('focus_auto', 0)
        self.set_property('focus_absolute', 0)

        mode = self.camera.getVideoMode()
        logging.info("camera '%s' pixel format = %s, %dx%d, %dFPS", name,
                     mode.pixelFormat, mode.width, mode.height, mode.fps)

        # Variables for the threaded read loop
        self.sink = camera_server.getVideo(camera=self.camera)

        self.calibration_matrix = None
        self.distortion_matrix = None

        self.threaded = threaded
        self.frametime = None
        self.camera_frame = None
        self.stopped = False
        self.frame_number = 0
        self.last_read = 0

        return

    def get_name(self):
        return self.camera.getName()

    def set_exposure(self, value):
        '''Set the camera exposure. 0 means auto exposure'''

        logging.info(f"Setting camera exposure to '{value}'")
        if value == 0:
            self.camera.setExposureAuto()
            # Logitech does not like having exposure_auto_priority on when the light is poor
            #  slows down the frame rate
            # camera.getProperty('exposure_auto_priority').set(1)
        else:
            self.camera.setExposureManual(int(value))
            # camera.getProperty('exposure_auto_priority').set(0)
        return

    def set_property(self, name, value):
        '''Set a camera property, such as auto_focus'''

        logging.info(f"Setting camera property '{name}' to '{value}'")
        try:
            try:
                propVal = int(value)
            except ValueError:
                self.camera.getProperty(name).setString(value)
            else:
                self.camera.getProperty(name).set(propVal)
        except Exception as e:
            logging.warn("Unable to set property '{}': {}".format(name, e))

        return

    def start(self):
        '''Start the thread to read frames from the video stream'''

        if self.threaded:
            t = Thread(target=self.update, args=())
            t.daemon = True
            t.start()
        return

    def update(self):
        '''Threaded read loop'''

        fps_startt = time.time()

        while True:
            # if the thread indicator variable is set, stop the thread
            if self.stopped:
                return

            # otherwise, read the next frame from the stream
            self._read_one_frame()

            if self.frame_number % 150 == 0:
                endt = time.time()
                dt = endt - fps_startt
                logging.info("threadedcamera: 150 frames in {0:.3f} seconds = {1:.2f} FPS".format(dt, 150.0 / dt))
                fps_startt = endt
        return

    def next_frame(self):
        '''Wait for a new frame'''

        if self.threaded:
            while self.last_read == self.frame_number:
                sleep(0.001)
            self.last_read = self.frame_number
        else:
            self._read_one_frame()

        return self.frametime, self.camera_frame

    def get_frame(self):
        '''Return the frame most recently read, no waiting. This may be a repeat of the previous image.'''

        if not self.threaded:
            raise Exception("Called get_frame on a non-threaded reader")

        return self.frametime, self.camera_frame

    def _read_one_frame(self):
        self.frametime, self.camera_frame = self.sink.grabFrame(self.camera_frame)
        self.frame_number += 1

        if self.rot90_count and self.frametime > 0:
            # Numpy is *much* faster than the OpenCV routine
            self.camera_frame = rot90(self.camera_frame, self.rot90_count)
        return

    def stop(self):
        # indicate that the thread should be stopped
        self.stopped = True
        sleep(0.3)         # time for thread to stop (proper way???)
        return


class LogitechC930e(Camera):
    def __init__(self, camera_server, name, device, height=240, fps=30, width=None, rotation=0, threaded=False):
        if not width:
            width = 424 if height == 240 else 848

        super().__init__(camera_server, name, device, height=height, fps=fps, width=width, rotation=rotation, threaded=threaded)

        # Logitech does not like having exposure_auto_priority on when the light is poor
        #  slows down the frame rate
        self.set_property('exposure_auto_priority', 0)

        return
        
class GenericFinder:
    def __init__(self, name, camera, finder_id=1.0, exposure=0, rotation=None, line_coords=None):
        self.name = name
        self.finder_id = float(finder_id)   # id needs to be float! "id" is a reserved word.
        self.camera = camera                # string with camera name
        self.stream_camera = None           # None means same camera
        self.exposure = exposure
        self.rotation = rotation            # cv2.ROTATE_90_CLOCKWISE = 0, cv2.ROTATE_180 = 1, cv2.ROTATE_90_COUNTERCLOCKWISE = 2
        self.line_coords = line_coords      # coordinates to draw a line on the image
        self.startAt = 0
        return

    def process_image(self, camera_frame):
        '''Main image processing routine'''

        # for 2020, standard result includes position, angle of a 2nd ball
        return (1.0, self.finder_id, 0.0, 0.0, 0.0, -1.0, -1.0)

    def prepare_output_image(self, input_frame):
        '''Prepare output image for drive station. Rotate image if needed, otherwise nothing to do.'''
        # WARNING rotation=0 is actually 90deg clockwise (dumb!!)
        if self.rotation is not None:
            # rotate function makes a copy, so no need to do that ahead.
            output_frame = cv2.rotate(input_frame, self.rotation)
        else:
            output_frame = input_frame.copy()
        if self.line_coords is not None:
            cv2.line(output_frame, self.line_coords[0], self.line_coords[1], (255, 255, 255), 2)

        return output_frame

    def set_start_time():

        self.startAt = datetime.timestamp(datetime.now())

    # ----------------------------------------------------------------------
    # the routines below are not needed here, but are used by lots of Finders,
    #  so keep them in one place

    # Uses trig and focal length of camera to find yaw.
    # Link to further explanation: https://docs.google.com/presentation/d/1ediRsI-oR3-kwawFJZ34_ZTlQS2SDBLjZasjzZ-eXbQ/pub?start=false&loop=false&slide=id.g12c083cffa_0_298
    @staticmethod
    def calculateYaw(pixelX, centerX, hFocalLength):
        yaw = math.degrees(math.atan((pixelX - centerX) / hFocalLength))
        return round(yaw)


    # Link to further explanation: https://docs.google.com/presentation/d/1ediRsI-oR3-kwawFJZ34_ZTlQS2SDBLjZasjzZ-eXbQ/pub?start=false&loop=false&slide=id.g12c083cffa_0_298
    @staticmethod
    def calculatePitch(pixelY, centerY, vFocalLength):
        pitch = math.degrees(math.atan((pixelY - centerY) / vFocalLength))
        # Just stopped working have to do this:
        pitch *= -1
        return round(pitch)

    @staticmethod
    def contour_center_width(contour):
        '''Find boundingRect of contour, but return center and width/height'''

        x, y, w, h = cv2.boundingRect(contour)
        return (x + int(w / 2), y + int(h / 2)), (w, h)

    @staticmethod
    def quad_fit(contour):
        '''Best fit of a quadrilateral to the contour'''

        approx = approxPolyDP_adaptive(contour, nsides=4)
        return hough_fit(contour, nsides=4, approx_fit=approx)

    @staticmethod
    def sort_corners(contour, center=None):
        '''Sort the contour in our standard order, starting upper-left and going counter-clockwise'''

        # Note: the inputs are all numpy arrays, so it is fast to operate on the whole array at once

        if center is None:
            center = contour.mean(axis=0)

        d = contour - center
        # remember that y-axis increases down, so flip the sign
        angle = (numpy.arctan2(-d[:, 1], d[:, 0]) - pi_by_2) % two_pi
        return contour[numpy.argsort(angle)]

    @staticmethod
    def major_minor_axes(moments):
        '''Compute the major/minor axes and orientation of an object from the moments'''

        # See https://en.wikipedia.org/wiki/Image_moment
        # Be careful, different sites define the normalized central moments differently
        # See also http://raphael.candelier.fr/?blog=Image%20Moments

        m00 = moments['m00']
        mu20 = moments['mu20'] / m00
        mu02 = moments['mu02'] / m00
        mu11 = moments['mu11'] / m00

        descr = math.sqrt(4.0 * mu11*mu11 + (mu20 - mu02)**2)

        major = math.sqrt(2.0 * (mu20 + mu02 + descr))
        minor = math.sqrt(2.0 * (mu20 + mu02 - descr))

        # note this does not use atan2.
        angle = 0.5 * math.atan(2*mu11 / (mu20-mu02))
        if mu20 < mu02:
            angle += pi_by_2

        return major, minor, angle

class CountDownFinder(GenericFinder):
    def __init__(self, name, camera, finder_id=6.0, exposure=0, rotation=None, line_coords=None):
        self.name = name
        self.finder_id = float(finder_id)   # id needs to be float! "id" is a reserved word.
        self.camera = camera                # string with camera name
        self.stream_camera = None           # None means same camera
        self.exposure = exposure
        self.rotation = rotation            # cv2.ROTATE_90_CLOCKWISE = 0, cv2.ROTATE_180 = 1, cv2.ROTATE_90_COUNTERCLOCKWISE = 2
        self.line_coords = line_coords      # coordinates to draw a line on the image
        return

    def prepare_output_image(self, input_frame):
        '''Prepare output image for drive station. Rotate image if needed, otherwise nothing to do.'''
        # WARNING rotation=0 is actually 90deg clockwise (dumb!!)
        if self.rotation is not None:
            # rotate function makes a copy, so no need to do that ahead.
            output_frame = cv2.rotate(input_frame, self.rotation)
        else:
            output_frame = input_frame.copy()
        if self.line_coords is not None:
            cv2.line(output_frame, self.line_coords[0], self.line_coords[1], (255, 255, 255), 2)

        current_dt = datetime.timestamp(datetime.now())
        dt = 40 - (current_dt - self.startAt)
        if dt < 0 :
            dt=0


        cv2.putText(output_frame, "{0:.1f} seconds".format(dt), (5, output_frame.shape[0]-5), cv2.FONT_HERSHEY_SIMPLEX,
                            0.75, (0, 255, 255), thickness=2)

        return output_frame

class BallFinder2020(GenericFinder):
    '''Ball finder for Infinite Recharge 2020'''

    HFOV = 64.0                  # horizontal angle of the field of view
    VFOV = 52.0                  # vertical angle of the field of view

    # create imaginary view plane on 3d coords to get height and width
    # place the view place on 3d coordinate plane 1.0 unit away from (0, 0) for simplicity
    VP_HALF_WIDTH = math.tan(math.radians(HFOV)/2.0)  # view plane 1/2 height
    VP_HALF_HEIGHT = math.tan(math.radians(VFOV)/2.0)  # view plane 1/2 width

    def __init__(self, CALIB_STRING, name='intake', camera='intake', finder_id=2.0, exposure=0):
        super().__init__(name=name, camera=camera, finder_id=finder_id, exposure=exposure)

        # Color threshold values, in HSV space
        self.low_limit_hsv = BALL_LOW_HSV
        self.high_limit_hsv =BALL_HIGH_HSV

        self.approx_polydp_error = 0.015
        # pixel area of the bounding rectangle - just used to remove stupidly small regions
        self.contour_min_area = 80

        # self.erode_kernel = numpy.ones((3, 3), numpy.uint8)
        # self.erode_iterations = 0

        # some variables to save results for drawing
        self.top_contours = []
        self.found_contours = []
        self.center_points = []

        self.cameraMatrix = None
        self.distortionMatrix = None
        if CALIB_STRING:
            json_data = json.loads(CALIB_STRING)
            self.cameraMatrix = numpy.array(json_data["camera_matrix"])
            self.distortionMatrix = numpy.array(json_data["distortion"])

        self.tilt_angle = math.radians(-7.5)  #FLOOR_CAMERA_ANGLE #  camera mount angle (radians)
        self.camera_height = 31 #15.50            # height of camera off the ground (inches)
        self.target_height = 0.0             # height of target off the ground (inches)

        return

    def set_color_thresholds(self, hue_low, hue_high, sat_low, sat_high, val_low, val_high):
        self.low_limit_hsv = numpy.array((hue_low, sat_low, val_low), dtype=numpy.uint8)
        self.high_limit_hsv = numpy.array((hue_high, sat_high, val_high), dtype=numpy.uint8)
        return

    def get_ball_values(self, center, shape):
        '''Calculate the angle and distance from the camera to the center point of the robot
        This routine uses the FOV numbers and the default center to convert to normalized coordinates'''

        # center is in pixel coordinates, 0,0 is the upper-left, positive down and to the right
        # (nx,ny) = normalized pixel coordinates, 0,0 is the center, positive right and up
        # WARNING: shape is (h, w, nbytes) not (w,h,...)
        image_w = shape[1] / 2.0
        image_h = shape[0] / 2.0

        # NOTE: the 0.5 is to place the location in the center of the pixel
        # print("center", center, "shape", shape)
        nx = (center[0] - image_w + 0.5) / image_w
        ny = (image_h - 0.5 - center[1]) / image_h

        # convert normal pixel coords to pixel coords
        x = BallFinder2020.VP_HALF_WIDTH * nx
        y = BallFinder2020.VP_HALF_HEIGHT * ny
        # print("values", center[0], center[1], nx, ny, x, y)

        # now have all pieces to convert to angle:
        ax = math.atan2(x, 1.0)     # horizontal angle

        # naive expression
        # ay = math.atan2(y, 1.0)     # vertical angle

        # corrected expression.
        # As horizontal angle gets larger, real vertical angle gets a little smaller
        ay = math.atan2(y * math.cos(ax), 1.0)     # vertical angle
        logging.info("ax, ay: {}, {}".format(math.degrees(ax), math.degrees(ay)))

        # now use the x and y angles to calculate the distance to the target:
        d = (self.target_height - self.camera_height) / math.tan(self.tilt_angle + ay)    # distance to the target

        return ax, d    # return horizontal angle and distance

    def get_ball_values_calib(self, center):
        '''Calculate the angle and distance from the camera to the center point of the robot
        This routine uses the cameraMatrix from the calibration to convert to normalized coordinates'''

        # use the distortion and camera arrays to correct the location of the center point
        # got this from
        #  https://stackoverflow.com/questions/8499984/how-to-undistort-points-in-camera-shot-coordinates-and-obtain-corresponding-undi

        ptlist = numpy.array([[center]])
        out_pt = cv2.undistortPoints(ptlist, self.cameraMatrix, self.distortionMatrix, P=self.cameraMatrix)
        undist_center = out_pt[0, 0]

        x_prime = (undist_center[0] - self.cameraMatrix[0, 2]) / self.cameraMatrix[0, 0]
        y_prime = -(undist_center[1] - self.cameraMatrix[1, 2]) / self.cameraMatrix[1, 1]

        # now have all pieces to convert to angle:
        ax = math.atan2(x_prime, 1.0)     # horizontal angle

        # naive expression
        # ay = math.atan2(y_prime, 1.0)     # vertical angle

        # corrected expression.
        # As horizontal angle gets larger, real vertical angle gets a little smaller
        ay = math.atan2(y_prime * math.cos(ax), 1.0)     # vertical angle
        logging.info("ax, ay: {}, {}".format(math.degrees(ax), math.degrees(ay)))

        # now use the x and y angles to calculate the distance to the target:
        d = (self.target_height - self.camera_height) / math.tan(self.tilt_angle + ay)    # distance to the target

        return ax, d    # return horizontal angle and distance

    def process_image_2877(self, camera_frame):
        '''Main image processing routine'''

        # clear out member result variables
        self.center_points = []
        self.top_contours = []
        self.found_contours = []

        camera_frame = cv2.medianBlur(camera_frame,5)
        hsv_frame = cv2.cvtColor(camera_frame, cv2.COLOR_BGR2HSV)
        threshold_frame = cv2.inRange(hsv_frame, self.low_limit_hsv, self.high_limit_hsv)

        # if self.erode_iterations > 0:
        #     erode_frame = cv2.erode(threshold_frame, self.erode_kernel, iterations=self.erode_iterations)
        # else:
        #     erode_frame = threshold_frame

        # OpenCV 3 returns 3 parameters!
        # Only need the contours variable
        _, contours, _ = cv2.findContours(threshold_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        contour_list = []
        for c in contours:
            center, widths = self.contour_center_width(c)
            area = widths[0] * widths[1]
            if area > self.contour_min_area:
                contour_list.append({'contour': c, 'center': center, 'widths': widths, 'area': area})

        self.top_contours = [cnt['contour'] for cnt in contour_list]
        # Sort the list of contours from lowest in the image to the highest in the image
        contour_list.sort(key=lambda c: c['center'][1], reverse=True)

        # test the lowest three contours only (optimization)
        for cnt in contour_list[0:3]:
            result_cnt = self.test_candidate_contour(cnt)
            if result_cnt is not None:
                self.found_contours.append(result_cnt)

                img_moments = cv2.moments(result_cnt)
                center = numpy.array((img_moments['m10']/img_moments['m00'], img_moments['m01']/img_moments['m00']))

                # note: these are the major/minor axes, equivalent to the radius (not the diameter)
                major, minor, angle = self.major_minor_axes(img_moments)

                # if the ratio is large, probably is 2 balls merged into 1 contour
                print("major/minor axis ratio", major / minor)
                if major / minor > 1.4:
                    # compute the offset, otherwise just use the centroid (more reliable)
                    direction = numpy.array((math.cos(angle), math.sin(angle)))

                    # the factor of 1.3 is total arbitrary, but seems to make the point closer to the center
                    # essentially the minor axis underestimates the radius of the front ball,
                    #  which is bigger in the image
                    self.center_points.append(center + ((major - 1.3 * minor) * direction))

                    # if the contour is made up of all three balls, must return 2 centers, so return both anyways
                    self.center_points.append(center - ((major - 0.8 * minor) * direction))

                    # TODO may need to change the 1.3 and 0.8 for three vs. two balls?
                else:
                    self.center_points.append(center)
                # print("Center point:", center)

                if len(self.center_points) >= 2:
                    break

        # done with the contours. Pick two centers to return

        # return values: (success, finder_id, distance1, robot_angle1, target_angle1, distance2, robot_angle2)
        # -1.0 means that something failed
        # 0 means that the entry is not currently being used

        if not self.center_points:
            # failed, found no ball
            return (0.0, self.finder_id, -1.0, -1.0, 0.0, -1.0, -1.0)

        # remember y goes up as you move down the image
        # self.center_points.sort(key=lambda c: c[1], reverse=True) #no need b/c it should already be in order

        if self.cameraMatrix is not None:
            # use the camera calibration if we have it
            angle1, distance1 = self.get_ball_values_calib(self.center_points[0])
            if len(self.center_points) > 1:
                angle2, distance2 = self.get_ball_values_calib(self.center_points[1])
            else:
                angle2 = -1.0
                distance2 = -1.0
        else:
            angle1, distance1 = self.get_ball_values(self.center_points[0], camera_frame.shape)
            if len(self.center_points) > 1:
                angle2, distance2 = self.get_ball_values(self.center_points[1], camera_frame.shape)
            else:
                angle2 = -1.0
                distance2 = -1.0

        return (1.0, self.finder_id, distance1, angle1, 0.0, distance2, angle2)

    def test_candidate_contour(self, contour_entry):
        cnt = contour_entry['contour']

        # real_area = cv2.contourArea(cnt)
        # print('areas:', real_area, contour_entry['area'], real_area / contour_entry['area'])
        # print("ratio"+str(contour_entry['widths'][1] / contour_entry['widths'][0] ))

        # contour_entry['widths'][1] is the height
        # contour_entry['widths'][0] is the width

        ratio = contour_entry['widths'][1] / contour_entry['widths'][0]
        # TODO if balls cut out at bottom of screen it returns none,
        #   so might want to change the lower value depending on camera location
        if ratio < 0.8 or ratio > 3.1:
            return None

        ratio = cv2.contourArea(cnt) / contour_entry['area']
        if ratio < (math.pi / 4) - 0.1 or ratio > (math.pi / 4) + 0.1:  # TODO refine the 0.1 error range
            return None

        return cnt

    def prepare_output_image_2877(self, input_frame):
        '''Prepare output image for drive station. Draw the found target contour.'''

        output_frame = input_frame.copy()

        # Draw the contour on the image
        # if self.top_contours:
        #     cv2.drawContours(output_frame, self.top_contours, -1, (255, 0, 0), 1)

        if len(self.found_contours) > 0:            
            ((x, y), radius) = cv2.minEnclosingCircle(self.found_contours[0])

            M = cv2.moments(self.found_contours[0])
            center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

            # only proceed if the radius meets a minimum size
            if radius > 10:
                # draw the circle and centroid on the frame,
                # then update the list of tracked points
                cv2.circle(output_frame, (int(x), int(y)), int(radius),
                    (0, 255, 255), 2)
                cv2.circle(output_frame, center, 5, (0, 0, 255), -1)
            # cv2.drawContours(output_frame, self.found_contours, -1, (0, 0, 255), 1)

        # for cnt in self.center_points:
        #     cv2.drawMarker(output_frame, tuple(cnt.astype(int)), (255, 125, 0), cv2.MARKER_CROSS, 5, 1)

        return output_frame

    def process_image_simple(self, camera_frame):
        '''Main image processing routine'''

        # Gets the shape of video
        screenHeight, screenWidth, _ = camera_frame.shape
        # Gets center of height and width
        centerX = (screenWidth / 2) - .5
        centerY = (screenHeight / 2) - .5

        # camera_frame = cv2.medianBlur(camera_frame,5)
        blurred = cv2.GaussianBlur(camera_frame, (27, 27), 0)
        hsv_frame = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

        threshold_frame = cv2.inRange(hsv_frame, self.low_limit_hsv, self.high_limit_hsv)

        # construct a mask for the color "yellow", then perform
        # a series of dilations and erosions to remove any small
        # blobs left in the mask
        # mask = cv2.erode(threshold_frame, None, iterations=2)
        # mask = cv2.dilate(mask, None, iterations=2)

        # OpenCV 3 returns 3 parameters!
        # Only need the contours variable
        _, contours, _ = cv2.findContours(threshold_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        self.contour_list = []
        for c in contours:
            # center, widths = self.contour_center_width(c)
            ((x, y), radius) = cv2.minEnclosingCircle(c)
            # area = widths[0] * widths[1]
            if radius > 8:
                # TODO: use a simple class? Maybe use "attrs" package?                
                distance = (KNOWN_DISTANCE + BALL_DIAMETER / 2) * KNOWN_RADIUS / radius - BALL_DIAMETER/2        # the radius of ball is 27.5 when distance is 48 inch
                
                finalTarget = self.calculateYaw(x, centerX, H_FOCAL_LENGTH)
                print("x, y, radius, distance, yaw: ", x, y, radius, distance, finalTarget)

                self.contour_list.append({'contour': c, 'x': x, 'y': y, 'radius': radius, 'distance': distance, 'yaw': finalTarget})

        # Sort the list of contours from biggest area to smallest
        self.contour_list.sort(key=lambda c: c['radius'], reverse=True)
        
        if len(self.contour_list) == 0:
            return (0.0, self.finder_id, 0.0, 0.0, 0.0)

        return (1.0, self.finder_id, self.contour_list[0]['distance'], self.contour_list[0]['yaw'], 0.0)

    def prepare_output_image_simple(self, input_frame):
        '''Prepare output image for drive station. Draw the found target contour.'''

        output_frame = input_frame.copy()
        if self.contour_list is not None:    
            idx = 1        
            for cnt in self.contour_list[0:3]:

                x= cnt["x"]
                y= cnt["y"]
                radius= cnt["radius"]
                yaw= cnt["yaw"]

                M = cv2.moments(cnt["contour"])
                center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

                # only proceed if the radius meets a minimum size
                if radius > 10:
                    # draw the circle and centroid on the frame,
                    # then update the list of tracked points
                    cv2.circle(output_frame, (int(x), int(y)), int(radius),
                        (0, 255, 255 * idx), 2)
                    cv2.circle(output_frame, center, 5, (0, 0, 255), -1)
                    if idx == 1:
                        print("x,y,radius: ", x, y, radius)
                        distance = (KNOWN_DISTANCE + BALL_DIAMETER / 2) * KNOWN_RADIUS / radius - BALL_DIAMETER/2        # the radius of ball is 27.5 when distance is 48 inch

                        print("distance, yaw: ", distance, yaw)
                        cv2.putText(output_frame, "%.2fin" % distance, (output_frame.shape[1] - 200, output_frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 255, 0), 3)
                    
                idx = 0
        return output_frame

    def process_image(self, camera_frame):
        '''Main image processing routine'''
        # r = self.process_image_simple(camera_frame)
        # print("Simple Result: ", r)
        return self.process_image_2877(camera_frame)


    def prepare_output_image(self, input_frame):
        '''Prepare output image for drive station. Draw the found target contour.'''
        
        # return self.prepare_output_image_simple(input_frame)
        return self.prepare_output_image_2877(input_frame)


class GoalFinder2020(GenericFinder):
    '''Find high goal target for Infinite Recharge 2020'''

    # real world dimensions of the goal target
    # These are the full dimensions around both strips
    TARGET_STRIP_LENGTH = 19.625    # inches
    TARGET_HEIGHT = 17.0            # inches@!
    TARGET_TOP_WIDTH = 39.25        # inches
    TARGET_BOTTOM_WIDTH = TARGET_TOP_WIDTH - 2*TARGET_STRIP_LENGTH*math.cos(math.radians(60))

    # [0, 0] is center of the quadrilateral drawn around the high goal target
    # [top_left, bottom_left, bottom_right, top_right]
    real_world_coordinates = numpy.array([
        [-TARGET_TOP_WIDTH / 2, TARGET_HEIGHT / 2, 0.0],
        [-TARGET_BOTTOM_WIDTH / 2, -TARGET_HEIGHT / 2, 0.0],
        [TARGET_BOTTOM_WIDTH / 2, -TARGET_HEIGHT / 2, 0.0],
        [TARGET_TOP_WIDTH / 2, TARGET_HEIGHT / 2, 0.0]
    ])

    def __init__(self, CALIB_STRING, name='shooter', camera='shooter', finder_id=1.0, exposure=1):
        super().__init__(name=name, camera=camera, finder_id=finder_id, exposure=exposure)

        # Color threshold values, in HSV space
        self.low_limit_hsv = GOAL_LOW_HSV
        self.high_limit_hsv = GOAL_HIGH_HSV

        # pixel area of the bounding rectangle - just used to remove stupidly small regions
        self.contour_min_area = 80

        # candidate cut thresholds
        self.min_dim_ratio = 1
        self.max_area_ratio = 0.25

        # camera mount angle (radians)
        # NOTE: not sure if this should be positive or negative
        self.tilt_angle = math.radians(SHOOTER_CAMERA_ANGLE)

        self.hsv_frame = None
        self.threshold_frame = None

        # DEBUG values
        self.top_contours = None

        # output results
        self.target_contour = None

        if CALIB_STRING:
            json_data = json.loads(CALIB_STRING)
            self.cameraMatrix = numpy.array(json_data["camera_matrix"])
            self.distortionMatrix = numpy.array(json_data["distortion"])

        self.outer_corners = []
        return

    def set_color_thresholds(self, hue_low, hue_high, sat_low, sat_high, val_low, val_high):
        self.low_limit_hsv = numpy.array((hue_low, sat_low, val_low), dtype=numpy.uint8)
        self.high_limit_hsv = numpy.array((hue_high, sat_high, val_high), dtype=numpy.uint8)
        return

    @staticmethod
    def get_outer_corners(cnt):
        '''Return the outer four corners of a contour'''

        return GenericFinder.sort_corners(cnt)  # Sort by x value of cnr in increasing value

    def preallocate_arrays(self, shape):
        '''Pre-allocate work arrays to save time'''

        self.hsv_frame = numpy.empty(shape=shape, dtype=numpy.uint8)
        # threshold_fame is grey, so only 2 dimensions
        self.threshold_frame = numpy.empty(shape=shape[:2], dtype=numpy.uint8)
        return

    def process_image(self, camera_frame):
        '''Main image processing routine'''
        self.target_contour = None

        # DEBUG values; clear any values from previous image
        self.top_contours = None
        self.outer_corners = None

        shape = camera_frame.shape
        if self.hsv_frame is None or self.hsv_frame.shape != shape:
            self.preallocate_arrays(shape)

        self.hsv_frame = cv2.cvtColor(camera_frame, cv2.COLOR_BGR2HSV, dst=self.hsv_frame)
        self.threshold_frame = cv2.inRange(self.hsv_frame, self.low_limit_hsv, self.high_limit_hsv,
                                           dst=self.threshold_frame)

        # OpenCV 3 returns 3 parameters!
        # Only need the contours variable
        _, contours, _ = cv2.findContours(self.threshold_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        contour_list = []
        for c in contours:
            center, widths = GoalFinder2020.contour_center_width(c)
            area = widths[0] * widths[1]
            if area > self.contour_min_area:        # area cut
                contour_list.append({'contour': c, 'center': center, 'widths': widths, 'area': area})

        # Sort the list of contours from biggest area to smallest
        contour_list.sort(key=lambda c: c['area'], reverse=True)

        # DEBUG
        self.top_contours = [x['contour'] for x in contour_list]

        # try only the 5 biggest regions at most
        target_center = None
        for candidate_index in range(min(5, len(contour_list))):
            self.target_contour = self.test_candidate_contour(contour_list[candidate_index], shape)
            if self.target_contour is not None:
                target_center = contour_list[candidate_index]['center']
                break

        if self.target_contour is not None:
            # The target was found. Convert to real world co-ordinates.

            # need the corners in proper sorted order, and as floats
            self.outer_corners = GenericFinder.sort_corners(self.target_contour, target_center).astype(numpy.float)

            # print("Outside corners: ", self.outer_corners)
            # print("Real World target_coords: ", self.real_world_coordinates)

            retval, rvec, tvec = cv2.solvePnP(self.real_world_coordinates, self.outer_corners,
                                              self.cameraMatrix, self.distortionMatrix)
            if retval:
                result = [1.0, self.finder_id, ]
                result.extend(self.compute_output_values(rvec, tvec))
                result.extend((-1.0, -1.0))
                return result

        # no target found. Return "failure"
        return [0.0, self.finder_id, 0.0, 0.0, 0.0, -1.0, -1.0]

    def prepare_output_image(self, input_frame):
        '''Prepare output image for drive station. Draw the found target contour.'''

        output_frame = input_frame.copy()

        if self.top_contours:
            cv2.drawContours(output_frame, self.top_contours, -1, (0, 0, 255), 2)

        if self.target_contour is not None:
            cv2.drawContours(output_frame, [self.target_contour.astype(int)], -1, (255, 0, 0), 1)

        if self.outer_corners is not None:
            for indx, cnr in enumerate(self.outer_corners):
                cv2.circle(output_frame, tuple(cnr.astype(int)), 4, (0, 255, 0), -1, lineType=8, shift=0)
                # cv2.putText(output_frame, str(indx), tuple(cnr.astype(int)), 0, .5, (255, 255, 255))

        return output_frame

    def test_candidate_contour(self, candidate, shape):
        '''Determine the true target contour out of potential candidates'''

        cand_width = candidate['widths'][0]
        cand_height = candidate['widths'][1]

        cand_dim_ratio = cand_width / cand_height
        if cand_dim_ratio < self.min_dim_ratio:
            return None
        cand_area_ratio = cv2.contourArea(candidate["contour"]) / (cand_width * cand_height)
        if cand_area_ratio > self.max_area_ratio:
            return None

        hull = cv2.convexHull(candidate['contour'])
        contour = self.quad_fit(hull)

        if contour is not None and len(contour) == 4:
            return contour

        return None

    def compute_output_values(self, rvec, tvec):
        '''Compute the necessary output distance and angles'''

        # The tilt angle only affects the distance and angle1 calcs

        x = tvec[0][0]
        z = math.sin(self.tilt_angle) * tvec[1][0] + math.cos(self.tilt_angle) * tvec[2][0]

        # distance in the horizontal plane between camera and target
        distance = math.sqrt(x**2 + z**2) / 1.07

        # horizontal angle between camera center line and target
        angle1 = math.atan2(x, z)

        rot, _ = cv2.Rodrigues(rvec)
        rot_inv = rot.transpose()
        pzero_world = numpy.matmul(rot_inv, -tvec)
        angle2 = math.atan2(pzero_world[0][0], pzero_world[2][0])

        return distance, angle1, angle2

   
class VisionServer:
    '''Base class for the VisionServer'''

    # NetworkTable parameters

    # this will be under , but the SendableChooser code does not allow full paths
    ACTIVE_MODE_KEY = "/vision/active_mode"
    nt_active_mode = ntproperty(ACTIVE_MODE_KEY, "shooter")

    # frame rate is pretty variable, so set this a fair bit higher than what you really want
    # using a large number for no limit
    output_fps_limit = ntproperty('/vision/output_fps_limit', OUTPUT_FPS_LIMIT,
                                  doc='FPS limit of frames sent to MJPEG server')

    # fix the TCP port for the main video, so it does not change with multiple cameras
    output_port = ntproperty('/vision/output_port', 1190,
                             doc='TCP port for main image output')

    # Operation modes. Force the value on startup.
    tuning = ntproperty('/vision/tuning', False, writeDefault=True,
                        doc='Tuning mode. Reads processing parameters each time.')

    # Logitech c930 are wide-screen cameras, so 320x180 has the biggest FOV
    image_width = ntproperty('/vision/width', IMAGE_WIDTH, writeDefault=False, doc='Image width')
    image_height = ntproperty('/vision/height', IMAGE_HEIGHT, writeDefault=False, doc='Image height')
    camera_fps = ntproperty('/vision/fps', CAMERA_FPS, writeDefault=False, doc='FPS from camera')

    image_writer_state = ntproperty('/vision/write_images', False, writeDefault=True,
                                    doc='Turn on saving of images')

    # Targeting info sent to RoboRio
    # Send the results as one big array in order to guarantee that the results
    #  all arrive at the RoboRio at the same time.
    # Value is (time, success, finder_id, distance, angle1, angle2) as a flat array.
    # All values are floating point (required by NT).
    target_info = ntproperty('/vision/target_info', 6 * [0.0, ],
                             doc='Packed array of target info: time, success, finder_id, distance, angle1, angle2')

    def __init__(self, initial_mode, test_mode=False):
        self.test_mode = test_mode
        # for processing stored files and no camera
        self.file_mode = False

        # time of each frame. Sent to the RoboRio as a heartbeat
        self.image_time = 0

        self.camera_server = cscore.CameraServer.getInstance()
        self.camera_server.enableLogging()

        self.cameras = {}
        self.active_camera = None

        self.create_output_stream()

        # Dictionary of finders. The key is the string "name" of the finder.
        self.target_finders = {}

        # Initial mode for start of match.
        # VisionServer switches to this mode after a second, to get the cameras initialized
        self.initial_mode = initial_mode

        # SendableChooser creates a dropdown chooser in ShuffleBoard
        # self.mode_chooser = SendableChooser()
        # SmartDashboard.putData(self.ACTIVE_MODE_KEY, self.mode_chooser)
        # self.mode_chooser_ctrl = ChooserControl(self.ACTIVE_MODE_KEY)

        # active mode. To be compared to the value from mode_chooser to see if it has changed
        self.active_mode = None

        self.curr_finder = None

        # rate limit parameters
        self.previous_output_time = time.time()
        self.camera_frame = None
        self.output_frame = None

        # Last error message (from cscore)
        self.error_msg = None

        # if asked, save image every 1/2 second
        # images are saved under the directory 'saved_images' in the current directory
        #  (ie current directory when the server is started)
        self.image_writer = ImageWriter(location_root='./saved_images',
                                        capture_period=0.5, image_format='jpg')

        return

    # --------------------------------------------------------------------------------
    # Methods generally customized each year

    """Methods you should/must include"""

    def update_parameters(self):
        '''Update processing parameters from NetworkTables values.
        Only do this on startup or if "tuning" is on, for efficiency'''

        # Make sure to add any additional created properties which should be changeable
        raise NotImplementedError

    # --------------------------------------------------------------------------------
    # Methods which hopefully don't need to be updated

    def preallocate_arrays(self):
        '''Preallocate the intermediate result image arrays'''

        # NOTE: shape is (height, width, #bytes)
        # self.camera_frame = numpy.zeros(shape=(int(self.image_height), int(self.image_width), 3),
        #                                 dtype=numpy.uint8)

        return

    def create_output_stream(self):
        '''Create the main image MJPEG server'''

        # Output server
        # Need to do this the hard way to set the TCP port
        self.output_stream = cscore.CvSource('camera', cscore.VideoMode.PixelFormat.kMJPEG, 416, 240, 25)
                                            #  int(self.image_width), int(self.image_height),
                                            #  int(min(self.camera_fps, self.output_fps_limit)))

        self.camera_server.addCamera(self.output_stream)
        server = self.camera_server.addServer(name='camera', port=int(self.output_port))
        server.setSource(self.output_stream)

        return

    def add_camera(self, camera, active=True):
        '''Add a single camera and set it to active/disabled as indicated.
        Cameras are referenced by their name, so pick something unique'''

        self.cameras[camera.get_name()] = camera
        camera.start()          # start read thread
        if active:
            self.active_camera = camera

        return

    def switch_camera(self, name):
        '''Switch the active camera, and disable the previously active one'''

        new_camera = self.cameras.get(name, None)
        if new_camera is not None:
            self.active_camera = new_camera
        else:
            logging.error('Unknown camera %s' % name)

        return

    def add_target_finder(self, finder):
        n = finder.name
        logging.info("Adding target finder '{}' id {}".format(n, finder.finder_id))
        self.target_finders[n] = finder
        # NetworkTables.getEntry(self.ACTIVE_MODE_KEY + '/options').setStringArray(self.target_finders.keys())

        # if n == self.initial_mode:
        #     NetworkTables.getEntry(self.ACTIVE_MODE_KEY + '/default').setString(n)
        #     self.mode_chooser_ctrl.setSelected(n)
        return        

    def switch_mode(self, new_mode):
        '''Switch processing mode. new_mode is the string name'''

        try:
            logging.info("Switching mode to '%s'" % new_mode)
            finder = self.target_finders.get(new_mode, None)
            if finder is not None:
                if self.active_camera.get_name() != finder.camera:
                    self.switch_camera(finder.camera)

                self.curr_finder = finder
                #finder.set_start_time()
                finder.startAt = datetime.timestamp(datetime.now())
                self.active_camera.set_exposure(finder.exposure)
                self.active_mode = new_mode
            else:
                logging.error("Unknown mode '%s'" % new_mode)

            # self.mode_chooser_ctrl.setSelected(self.active_mode)  # make sure they are in sync
        except Exception as e:
            logging.error('Exception when switching mode: %s', e)

        return

    def process_image(self):
        # rvec, tvec return as None if no target found
        try:
            result = self.curr_finder.process_image(self.camera_frame)
        except Exception as e:
            logging.error("Exception caught in process_image(): %s", e)
            result = (0.0, 0.0, 0.0, 0.0, 0.0)

        return result

    def prepare_output_image(self):
        '''Create the image to send to the Driver station.
        Finder is expected to *copy* the input image, as needed'''

        if self.camera_frame is None:
            return

        try:
            if self.curr_finder is None:
                self.output_frame = self.camera_frame.copy()
            else:
                base_frame = self.camera_frame

                # Finders are allowed to designate a different image to stream to the DS
                cam = self.curr_finder.stream_camera
                if cam is not None:
                    rdr = self.video_readers.get(cam, None)
                    if rdr is not None:
                        _, base_frame = rdr.get_frame()  # does not wait

                self.output_frame = self.curr_finder.prepare_output_image(base_frame)

            min_dim = min(self.output_frame.shape[0:2])

            # Rescale if needed
            if min_dim > 400:
                # downscale by 2x
                self.output_frame = cv2.resize(self.output_frame, dsize=(0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_NEAREST)
                min_dim //= 2

            if min_dim < 400:  # test on height
                dotrad = 3
                fontscale = 0.4
                fontthick = 1
            else:
                dotrad = 5
                fontscale = 0.75
                fontthick = 2

            # If saving images, add a little red "Recording" dot in upper left
            if self.image_writer_state:
                cv2.circle(self.output_frame, (20, 20), dotrad, (0, 0, 255), thickness=2*dotrad, lineType=8, shift=0)

            # If tuning mode is on, add text to the upper left corner saying "Tuning On"
            if self.tuning:
                cv2.putText(self.output_frame, "TUNING ON", (60, 30), cv2.FONT_HERSHEY_SIMPLEX, fontscale, (255, 0, 0), thickness=fontthick)

            # If test mode (ie running the NT server), give a warning
            if self.test_mode:
                cv2.putText(self.output_frame, "TEST MODE", (5, self.output_frame.shape[0]-5), cv2.FONT_HERSHEY_SIMPLEX,
                            fontscale, (0, 255, 255), thickness=fontthick)

        except Exception as e:
            logging.error("Exception caught in prepare_output_image(): %s", e)

        return

    def run(self):
        '''Main loop. Read camera, process the image, send to the MJPEG server'''

        frame_num = 0
        errors = 0

        fps_count = 0
        fps_startt = time.time()
        imgproc_nettime = 0

        # NetworkTables.addEntryListener(self.active_mode_changed)
        old_ts = datetime.timestamp(datetime.now())
        mode_idx=0
        mode = ["shooter","intake","arm","fisheye"]# "intake", "ballfinder", , "c930e"
        mode_period = [20,20,20,20]


        while True:
            try:
                # Check whether DS has asked for a different camera

                timestamp = datetime.timestamp(datetime.now())
                if timestamp - old_ts > mode_period[(mode_idx -1 ) % 4] :
                    self.switch_mode(mode[mode_idx % 4])
                    mode_idx += 1
                    old_ts = timestamp

                nt_mode = self.nt_active_mode

                # if nt_mode != self.active_mode:
                #     self.switch_mode(nt_mode)

                if self.tuning:
                    self.update_parameters()

                # Tell the CvReader to grab a frame from the camera and put it
                # in the source image.  Frametime==0 on error
                frametime, self.camera_frame = self.active_camera.next_frame()
                frame_num += 1

                imgproc_startt = time.time()

                if frametime == 0:
                    # ERROR!!
                    self.error_msg = self.active_camera.sink.getError()

                    if errors < 10:
                        errors += 1
                    else:   # if 10 or more iterations without any stream, switch cameras
                        logging.warning(self.active_camera.get_name() + " camera is no longer streaming. Switching cameras...")
                        # self.switch_mode(self.mode_after_error())
                        errors = 0

                    target_res = [time.time(), ]
                    target_res.extend(5*[0.0, ])
                else:
                    self.error_msg = None
                    errors = 0

                    if self.image_writer_state:
                        self.image_writer.setImage(self.camera_frame)

                    # frametime = time() * 1e8  (ie in 1/100 microseconds)
                    # convert frametime to seconds to use as the heartbeat sent to the RoboRio
                    target_res = [1e-8 * frametime, ]
                    
                    proc_result = self.process_image()                    
                    if proc_result[2] != 0 and proc_result[2] != -1:
                        logging.info("Process image result '{}' ".format(proc_result))
                    target_res.extend(proc_result)

                # Send the results as one big array in order to guarantee that the results
                #  all arrive at the RoboRio at the same time
                # Value is (Timestamp, Found, Mode, distance, angle1, angle2) as a flat array.
                #  All values are floating point (required by NT).
                self.target_info = target_res

                # Try to force an update of NT to the RoboRio. Docs say this may be rate-limited,
                #  so it might not happen every call.
                NetworkTables.flush()

                # Done. Output the marked up image, if needed
                # Note this rate limiting can also be done via the URL, but this is more efficient
                now = time.time()
                deltat = now - self.previous_output_time
                min_deltat = 1.0 / self.output_fps_limit
                if deltat >= min_deltat:
                    self.prepare_output_image()
                    self.output_stream.putFrame(self.output_frame)
                    self.previous_output_time = now

                if frame_num == 30:
                    # This is a bit stupid, but you need to poke the camera *after* the first
                    #  bunch of frames has been collected.
                    self.switch_mode(self.initial_mode)

                fps_count += 1
                imgproc_nettime += now - imgproc_startt
                if fps_count >= 150:
                    endt = time.time()
                    dt = endt - fps_startt
                    # logging.info("{0} frames in {1:.3f} seconds = {2:.2f} FPS".format(fps_count, dt, fps_count/dt))
                    # logging.info("Image processing time = {0:.2f} msec/frame".format(1000.0 * imgproc_nettime / fps_count))
                    fps_count = 0
                    fps_startt = endt
                    imgproc_nettime = 0

            except Exception as e:
                # major exception. Try to keep going
                logging.error('Caught general exception: %s', e)

        return

    def run_files(self, file_list):
        '''Run routine to loop through a set of files and process each.
        Waits a couple seconds between each, and loops forever'''

        self.file_mode = True
        file_index = 0
        while True:
            if self.camera_frame is None:
                self.preallocate_arrays()

            image_file = file_list[file_index]
            print('Processing', image_file)
            file_frame = cv2.imread(image_file)
            numpy.copyto(self.camera_frame, file_frame)

            self.process_image()

            self.prepare_output_image()

            self.output_stream.putFrame(self.output_frame)
            # probably don't want to use sleep. Want something thread-compatible
            # for _ in range(4):
            time.sleep(0.5)

            file_index = (file_index + 1) % len(file_list)
        return

# -----------------------------------------------------------------------------
def wait_on_nt_connect(max_delay=10):
    cnt = 0
    while True:
        if NetworkTables.isConnected():
            logging.info('Connect to NetworkTables after %d seconds', cnt)
            return

        if cnt >= max_delay:
            break

        if cnt > 0 and cnt % 5 == 0:
            logging.warning("Still waiting to connect to NT (%d sec)", cnt)
        time.sleep(1)
        cnt += 1

    logging.warning("Failed to connect to NetworkTables after %d seconds. Continuing", cnt)
    return

# syntax checkers don't like global variables, so use a simple function
def main(server_type):
    '''Main routine'''

    import argparse
    parser = argparse.ArgumentParser(description='2018 Vision Server')
    parser.add_argument('--calib', help='Calibration file for camera')
    parser.add_argument('--test', action='store_true', help='Run in local test mode')
    parser.add_argument('--delay', type=int, default=0, help='Max delay trying to connect to NT server (seconds)')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose. Turn up debug messages')
    parser.add_argument('--files', action='store_true', help='Process input files instead of camera')
    parser.add_argument('input_files', nargs='*', help='input files')

    args = parser.parse_args()

    # To see messages from networktables, you must setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format='%(asctime)s %(levelname)s: %(message)s')

    logging.info("cscore version '%s'" % cscore.__version__)
    logging.info("OpenCV version '%s'" % cv2.__version__)

    if args.test:
        # FOR TESTING, set this box as the server
        NetworkTables.enableVerboseLogging()
        NetworkTables.startServer()
    else:
        if args.verbose:
            # Turn up the noise from NetworkTables. VERY noisy!
            # DO NOT do this during competition, unless you are really sure
            NetworkTables.enableVerboseLogging()
        # NetworkTables.startClient('10.28.77.2')
        # Try startClientTeam() method; it auto tries a whole bunch of standard addresses
        NetworkTables.startClientTeam(7520)
        if args.delay > 0:
            wait_on_nt_connect(args.delay)
    server = server_type(CALIB_STRING=CALIB_STRING, test_mode=args.test)

    if args.files:
        if not args.input_files:
            parser.usage()

        server.run_files(args.input_files)
    else:
        server.run()
    return

class VisionServer2020(VisionServer):
    ACTIVE_MODE_KEY = "/vision/active_mode"

    # Retro-reflective target finding parameters

    # rrtarget_exposure = ntproperty('/vision/rrtarget/exposure', 0, doc='Camera exposure for rrtarget (0=auto)')

    def __init__(self, CALIB_STRING, test_mode=False):
        super().__init__(initial_mode='shooter', test_mode=test_mode)

        self.add_cameras()

        self.generic_fisheye = GenericFinder("fisheye", "fisheye", finder_id=4.0)
        self.add_target_finder(self.generic_fisheye)

        self.generic_arm = CountDownFinder("arm", "arm", finder_id=5.0)
        self.add_target_finder(self.generic_arm)

        cam = self.cameras['shooter']
        self.goal_finder = GoalFinder2020(CALIB_STRING) #C930E_CALIB_STRING
        self.add_target_finder(self.goal_finder)

        cam = self.cameras['intake']
        self.ball_finder = BallFinder2020(C930E_CALIB_STRING)
        self.add_target_finder(self.ball_finder)

        self.update_parameters()

        # start in intake mode to get cameras going. Will switch to 'shooter' after 1 sec.
        self.switch_mode('shooter')
        return

    def update_parameters(self):
        '''Update processing parameters from NetworkTables values.
        Only do this on startup or if "tuning" is on, for efficiency'''

        # Make sure to add any additional created properties which should be changeable

        # self.goal_finder.set_color_thresholds(65, 100,
        #                                       75, 255,
        #                                       15, 255)
        return
    def add_cameras(self):
        '''Add the cameras'''
        
        self.camera_device_1 = '/dev/v4l/by-path/platform-fd500000.pcie-pci-0000:01:00.0-usb-0:1.1:1.0-video-index0'    # for line and hatch processing
        self.camera_device_2 = '/dev/v4l/by-path/platform-fd500000.pcie-pci-0000:01:00.0-usb-0:1.2:1.0-video-index0'    # for line and hatch processing
        self.camera_device_3 = '/dev/v4l/by-path/platform-fd500000.pcie-pci-0000:01:00.0-usb-0:1.3:1.0-video-index0'    # for line and hatch processing
        self.camera_device_4 = '/dev/v4l/by-path/platform-fd500000.pcie-pci-0000:01:00.0-usb-0:1.4:1.0-video-index0'    # for line and hatch processing
        
        cam = Camera(self.camera_server, 'shooter', self.camera_device_2, width=424, height=240)
        self.add_camera(cam, True)

        cam = LogitechC930e(self.camera_server, 'intake', self.camera_device_4, width=424, height=240)
        self.add_camera(cam, False)

        cam = Camera(self.camera_server, 'arm', self.camera_device_1, width=424, height=240)
        self.add_camera(cam, False)

        cam = Camera(self.camera_server, 'fisheye', self.camera_device_3, width=640, height=480)
        self.add_camera(cam, False)
        return

# Main routine
if __name__ == '__main__':
    main(VisionServer2020)
