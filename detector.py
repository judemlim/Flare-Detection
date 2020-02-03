import sys
import math
import numpy as np
import cv2

def _calc_white_pixel_ratio(bw_image):
    '''
    Calculates the number of white pixels to black pixels on a binary image
    as a fraction between 0 and 1
    '''
    white_pixel_count = cv2.countNonZero(bw_image)
    total_pixel_count = len(bw_image)*len(bw_image[0])
    white_percentage = white_pixel_count / total_pixel_count
    return white_percentage

def _calculate_angle(numpy_line):
    '''
    Given a numpy line returns an angle in degrees
    '''
    x1, y1, x2, y2 = numpy_line[0]
    rise = y2 - y1
    run = x2 - x1
    if run == 0:
        return None

    angle_rad = math.atan(rise/run)
    return math.degrees(angle_rad)

def is_flare_lots(img, show_final_img_analysed=False):
    '''
    Checks what percentage of the image is light, and rejects if the percentage is
    too high.

    Returns a boolean value
    '''
    PERCENTAGE_THRESHOLD = 0.15

    grey_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    (_, bw_image) = cv2.threshold(grey_img, 240, 255, cv2.THRESH_BINARY)

    if show_final_img_analysed:
        cv2.imwrite("flare_lots.JPG", bw_image)

    return _calc_white_pixel_ratio(bw_image) > PERCENTAGE_THRESHOLD

def is_flare_elliptical(img, show_final_img_analysed=False):
    '''
    Looks for elliptical flares.

    Returns a boolean value.
    '''
    grey_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    (_, bw_image) = cv2.threshold(grey_img, 220, 255, cv2.THRESH_BINARY)
    params = cv2.SimpleBlobDetector_Params() 
  
    # Set Area filtering parameters 
    params.filterByArea = True
    params.minArea = 100
      
    # Set Circularity filtering parameters 
    params.filterByCircularity = True 
    params.minCircularity = 0.5
      
    # Set Convexity filtering parameters 
    params.filterByConvexity = True
    params.minConvexity = 0.95
          
    # Set inertia filtering parameters 
    params.filterByInertia = True
    params.minInertiaRatio = 0.2
      
    # Create a detector with the parameters 
    detector = cv2.SimpleBlobDetector_create(params) 
          
    # Detect blobs 
    inverted_img = cv2.bitwise_not(bw_image)
    keypoints = detector.detect(inverted_img) 
    
    if show_final_img_analysed:
        blank = np.zeros((1, 1))  
        blobs = cv2.drawKeypoints(inverted_img, keypoints, blank, (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS) 
          
        number_of_blobs = len(keypoints) 
        text = "Number of Circular Blobs: " + str(len(keypoints)) 
        cv2.putText(blobs, text, (20, 550),cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 100, 255), 2) 
          
        # Show blobs 
        cv2.imwrite("flare_elliptical.JPG", blobs)

    return True if keypoints else False

def is_flare_rays(img, show_final_img_analysed=False):
    '''
    GOAL: Looks for white straight lines of light in flared up images. If diagonl white lines are detect-
    ed then it is assumed that there is flare lensing

    Returns a boolean value.
    '''

    draft_img = img.copy()
    grey_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    (_, bw_image) = cv2.threshold(grey_img, 220, 255, cv2.THRESH_BINARY)
    interval1 = 30
    interval2 = 60

    # if there isn't much light in the image, relax flare detection criteria
    if _calc_white_pixel_ratio(bw_image) < 0.01:
        (_, bw_image) = cv2.threshold(grey_img, 180, 255, cv2.THRESH_BINARY)
        interval1 = 15
        interval2 = 75

    lines = cv2.HoughLinesP(bw_image, 1, np.pi/180, 100, 10000, 80)

    count = 0

    if lines is not None:
        for line in lines:
            angle = _calculate_angle(line)
            if angle:
                if angle >= interval1 and angle <= interval2:
                    count += 1
                elif angle <= interval1*-1 and angle >= interval2*-2:
                    count += 1

    if show_final_img_analysed:
        if lines is not None:
            for line in lines:
                for x1, y1, x2, y2 in line:
                    cv2.line(draft_img, (x1,y1),(x2,y2),(0,255,0),2)

        cv2.imwrite("flare_rays_bw.JPG", bw_image)
        cv2.imwrite("flare_rays_final.JPG", draft_img)

    return count

def is_flare_arcs(img, show_final_img_analysed=False):
    '''
    GOAL: Looks for an arc on a heavily eroded binary image. The arcs are
    formed from the flare disrupting the continuity of the wall-sky boundary.

    Returns a boolean value.

    UNSUCCESSFULL: Unable to fit an arc on the binary image. The flare induced arc
    proved to be too "messy" to approximate an actual circle on.
    '''
    grey_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    grey_img = cv2.blur(grey_img,(20,20)) 
    (_, bw_image) = cv2.threshold(grey_img, 240, 255, cv2.THRESH_BINARY)

    bw_image = cv2.erode(bw_image, None, iterations=6)

    circles = cv2.HoughCircles(bw_image, cv2.HOUGH_GRADIENT, 1.1, 100)

    if show_final_img_analysed:
        if circles is not None:
            # convert the (x, y) coordinates and radius of the circles to integers
            circles = np.round(circles[0, :]).astype("int")
            # loop over the (x, y) coordinates and radius of the circles
            for (x, y, r) in circles:
                # draw the circle in the output image, then draw a rectangle
                # corresponding to the center of the circle
                cv2.circle(img, (x, y), r, (0, 255, 0), 4)
                cv2.rectangle(img, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)
        
        cv2.imwrite("flare_arcs_bw.JPG", bw_image)
        cv2.imwrite("flare_arcs_final.JPG", img)

    return False if circles is None else True

if __name__ == "__main__":
    img_list = sys.argv[1:]
    flares = 0
    good = 0
    for img_name in img_list:
        img = cv2.imread(img_name)
        if is_flare_lots(img):
            print("1")
            flares += 1
        elif is_flare_elliptical(img):
            print("1")
            flares += 1
        elif is_flare_rays(img):
            print("1")
            flares += 1
    # Attempt to implement arcs due to flares was unsuccessful
        # elif is_flare_arcs(img):
            # print("1 - arcs")
        else:
            print("0")
            good += 1
    # print(f"flares - {flares}, good - {good}")

