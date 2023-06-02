
import numpy as np
import cv2

def findEdges(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask_green = cv2.inRange(hsv, (36, 25, 25), (70, 255, 255))
    edgesGreen = cv2.Canny(mask_green, 100, 255)
    edgesGreen[64:78, 44:52] = 0
    edgesGreen[83:-1, :] = 0
    kernel = np.ones((3, 3), np.uint8)
    edgesGreen = cv2.dilate(edgesGreen, kernel, iterations=2)
    edgesGreen = cv2.erode(edgesGreen, kernel, iterations=2)
    return edgesGreen

def findClosestEdgePos(edges, carPos = np.array([70, 48])):
    edgesPos = np.nonzero(edges)
    #Find single closest edge point
    distanceCarToEdges = np.linalg.norm(np.array(carPos)[:, None] - np.array(edgesPos), axis=0)
    closestEdgeIdx = np.argmin(distanceCarToEdges)
    closestEdgePos = np.array([edgesPos[0][closestEdgeIdx], edgesPos[1][closestEdgeIdx]])
    return closestEdgePos

def findTrackVector(edges, closestEdgePos):
    #Find vector describing track direction (tangent to track) using a square around the closest edge point
    squareSize = 3
    squareMiddlePoint = closestEdgePos
    square = edges.copy()[squareMiddlePoint[0] - squareSize: squareMiddlePoint[0] + squareSize + 1,
                                squareMiddlePoint[1] - squareSize: squareMiddlePoint[1] + squareSize + 1]
    edgesIdxSquare = np.nonzero(square)
    pnt1 = np.array([edgesIdxSquare[0][0], edgesIdxSquare[1][0]])
    pnt2 = np.array([edgesIdxSquare[0][-1], edgesIdxSquare[1][-1]])
    vector_track = pnt2 - pnt1
    return vector_track

def calculateTargetPoint(image, widthOfTrack, freq, scale_dist, Amplitude, t):
    # Find edges of track
    edges = findEdges(image) # returns a binary image with edges of track
    closestEdgePos = findClosestEdgePos(edges) # returns the position of the closest edge point to the car
    vector_track = findTrackVector(edges, closestEdgePos) # returns a vector describing the direction of the track
    
    #Make sure the track vector is pointing towards the car direction
    if np.dot(vector_track, np.array([-1, 0])) < 0:
        vector_track = -vector_track

    #Normalized track heading vector and perpendicular vector
    vector_track_normalized = vector_track / np.linalg.norm(vector_track)
    vector_track_perp_normalized = np.array([-vector_track_normalized[1], vector_track_normalized[0]])

    #Make sure that both vectors have reasonable values
    if np.isnan(vector_track_normalized).any() or np.isnan(vector_track_perp_normalized).any():
        return None, None, None, None

    #Check if the vector is pointing towards the inside of the track
    controlPixelPos = closestEdgePos + (vector_track_perp_normalized*3).astype(int)
    controlPixel = image[controlPixelPos[0], controlPixelPos[1]]
    if controlPixel[1] > 200: # Green pixel meaning outside of track
        vector_track_perp_normalized = -vector_track_perp_normalized
    
    #Find the estimated middle point of the track relative to the closest edge point
    estimatedMiddlePoint = (closestEdgePos + vector_track_perp_normalized * widthOfTrack / 2).astype(int)

    # Calculate the next num_points points on the trajectory (sinusoidal curve)
    sin_coeff = Amplitude * np.sin((t+1) * freq * 2 * np.pi)
    #cross product btw track vector and perpendicular vector positive
    sin_vector = (sin_coeff * vector_track_perp_normalized).astype(int)
    if np.cross(vector_track_normalized, vector_track_perp_normalized) < 0:
        sin_vector = -sin_vector
    sin_vector = sin_vector.astype(int)
    dir_vector = vector_track_normalized * scale_dist
    sinusPoints_pos = (estimatedMiddlePoint + dir_vector + sin_vector)
    targetPoint = sinusPoints_pos
    targetPoint = int(targetPoint[0]), int(targetPoint[1]) 

    return targetPoint, estimatedMiddlePoint, vector_track_normalized ,vector_track_perp_normalized

def controllerSteeringAngle(angle, prevSteeringAngle):
    #PID control of steering angle
    Kp = 0.5
    Ki = 0.1    
    Kd = -0.2
    steeringAngle = Kp * angle + Kd * (angle - prevSteeringAngle)
    return steeringAngle

def controllerSpeed(targetSpeed, currentSpeed, prevSpeedError):
    Kp = 0.5
    Ki = 0.1
    Kd = 0.1
    error = targetSpeed - currentSpeed
    acceleration = Kp * error + Kd * (error - prevSpeedError)
    breaking = 0
    prevSpeedError = error

    if acceleration < 0 :
        breaking = -acceleration
        acceleration = 0 
    return acceleration, breaking
