import cv2
import numpy as np

# constants
IMAGE_SIZE = 2000.0
MATCH_THRESHOLD = 40

# load haar cascade and street image
roundabout_cascade = cv2.CascadeClassifier('haarcascade_roundabout.xml')
street = cv2.imread('blue.jpg',1)
test = cv2.imread('blue.jpg',0)
kernel = np.ones((10,10),np.float32)/(10*10)
mapp = cv2.filter2D(street,-1,kernel)
# do roundabout detection on street image
gray = cv2.cvtColor(mapp,cv2.COLOR_BGR2HSV)
gray2 = cv2.cvtColor(street,cv2.COLOR_BGR2GRAY)
#cv2.imshow('street image', cv2.resize(gray, (0,0), fx=0.1, fy=0.1) )

    # define range of blue color in HSV
lower_blue = np.array([110,50,50])
upper_blue = np.array([130,255,255])

# Threshold the HSV image to get only blue colors
mask = cv2.inRange(gray, lower_blue, upper_blue)

# Copy the thresholded image.
im_floodfill = mask.copy()

# Mask used to flood filling.
# Notice the size needs to be 2 pixels than the image.
h, w = mask.shape[:2]
mask2 = np.zeros((h+2, w+2), np.uint8)

# Floodfill from point (0, 0)
cv2.floodFill(im_floodfill, mask2, (0,0), 255);

# Invert floodfilled image
im_floodfill_inv = cv2.bitwise_not(im_floodfill)

# Combine the two images to get the foreground.
mask = mask | im_floodfill_inv
# Bitwise-AND mask and original image
res = cv2.bitwise_and(street,street, mask= mask)


el = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
#image = cv2.dilate(mask, el, iterations=6)


im2, contours, hierarchy = cv2.findContours(
    mask,
    cv2.RETR_TREE,
    cv2.CHAIN_APPROX_SIMPLE
)
bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)
# find the keypoints and descriptors for roadsign image
roadsign = cv2.imread('left.jpg',0)
surf = cv2.xfeatures2d.SURF_create(400)
kp_r, des_r = surf.detectAndCompute(roadsign, None)
centers = []
radii = []
for contour in contours:
    area = cv2.contourArea(contour)
    # there is one contour that contains all others, filter it out
    if area < 10000:
        continue



    br = cv2.boundingRect(contour)
    radii.append(br[2])

    m = cv2.moments(contour)

    center = (int(m['m10'] / m['m00']), int(m['m01'] / m['m00']))
    centers.append(center)
    x,y,w,h = br
    print x
    # obtain object from street image
    obj = test[y:y+h,x:x+w]
    ratio = IMAGE_SIZE / obj.shape[1]

    obj = cv2.resize(obj,(int(IMAGE_SIZE),int(obj.shape[0]*ratio)))
    kp_o, des_o = surf.detectAndCompute(obj, None)
    if(len(des_o) < 5) : continue
    print len(des_o)
    matches = bf.match(des_r,des_o)
    print len(matches)
    img3 = cv2.drawMatches(roadsign,kp_r,obj,kp_o,matches[:50],None, flags=2)
    cv2.imshow(str(x),cv2.resize(img3, (0,0), fx=0.2, fy=0.2) )
cv2.drawContours(street, contours, -1, (0,255,0), 3)
print("There are {} circles".format(len(centers)))

radius = int(np.average(radii));

for center in centers:
    cv2.circle(street, center, 30, (255, 0, 0), -1)
    cv2.circle(street, center, radius, (255, 255, 0), 1)


cv2.imshow('frame',cv2.resize(street, (0,0), fx=0.2, fy=0.2) )
cv2.imshow('mask',cv2.resize(mask, (0,0), fx=0.2, fy=0.2) )
cv2.imshow('res',cv2.resize(res, (0,0), fx=0.2, fy=0.2) )
while(1):
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break
cv2.destroyAllWindows()
