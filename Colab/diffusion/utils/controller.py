
import numpy as np
import cv2

class Controller:
    def __init__(self):
        self.rgb_image = []
        self.TargetVelocity = 50
        
        # PD controller variables for angular control
        self.kp_ang = 0.5
        self.kd_ang = -0.01
        self.past_error_ang = 0

        # PD controller variables for angular control
        self.kp_dist = 0.01
        self.kd_dist = -0.005
        self.past_error_dist = 0
        
        # PD controller variables for velocity control
        self.kp_vel = 0.05
        self.kd_vel = 0.01
        self.past_error_vel = 0
    
    @staticmethod    
    def find_edge_1dStrip(array, direction):
        starting_point = int(len(array) // 2)

        idx = -1
        if direction == 'left':
            for i in range(starting_point, -1, -1):
                if array[i] != 0:
                    idx = i
                    break
        elif direction == 'right':
            for i in range(starting_point, len(array)):
                if array[i] != 0:
                    idx = i
                    break
        return idx
    
    def find_middle_point(self, strip_1d):
        idx1 = self.find_edge_1dStrip(strip_1d, 'left')
        idx2 = self.find_edge_1dStrip(strip_1d, 'right')

        if idx1 == -1:
            idx1 = 0
        if idx2 == -1:
            idx2 = len(strip_1d) - 1

        idx_middle = int((idx1 + idx2) / 2)
        return idx1, idx2, idx_middle
    

        
    
    def ProcessImage(self, obs):
        """ Input: 
                obs: image of the track
            Output:
                gray_mask: Strip of binary image  around the car position with black track an white background"""
        ## Cropping image
        strip_height = 20
        middle_height = 65
        top = int(middle_height - strip_height / 2)
        bottom = int(middle_height + strip_height / 2)
        # Crop the strip from the image
        strip = obs[top:bottom, :]
        ## Mask
        hsv = cv2.cvtColor(strip, cv2.COLOR_BGR2HSV)
        mask_green = cv2.inRange(hsv, (36, 25, 25), (70, 255,255))
        #slice the green
        imask_green = mask_green>0
        gray_mask = imask_green.astype(np.uint8)
        gray_mask = gray_mask*255

        strip_height, strip_width = gray_mask.shape
        upper_edge = gray_mask[0, :]
        lower_edge = gray_mask[strip_height - 1, :]
        _, _, idx_middle_upper = self.find_middle_point(upper_edge)
        _, _, idx_middle_lower = self.find_middle_point(lower_edge)

        # Compute error and angle using PID controllers
        distance_to_middle = strip_width // 2 - idx_middle_lower
        # Compute angular error
        dist_upper = strip_width // 2 - idx_middle_upper
        angular_error = np.arctan(dist_upper / strip_height)

        return distance_to_middle, angular_error


    def AngularPDController(self, error, past_error):
        # Calculate angular control output using PD controller
        control_output = self.kp_ang * error + self.kd_ang * (error - past_error)
        
        # Update past error for next iteration
        self.past_error_ang = error
        return -control_output

    def DistPDController(self, error, past_error):
          # Calculate angular control output using PD controller
          control_output = self.kp_dist * error + self.kd_dist * (error - past_error)
          
          # Update past error for next iteration
          self.past_error_dist = error
          return -control_output
    
    def VelocityPDController(self, vel_error, past_vel_error):
        # Calculate velocity control output using PD controller
        control_output = self.kp_vel * vel_error + self.kd_vel * (vel_error - past_vel_error)
        
        # Update past velocity error for next iteration
        self.past_error_vel = vel_error

        if control_output > 0:
            return (control_output, 0)
        else:
            return (0, -control_output)

    def calculateAction(self, rgb_image, velocity):
        # Get distance from processed image
        error_dist, error_ang = self.ProcessImage(rgb_image)
        
        # Calculate error for velocity control
        error_vel = self.TargetVelocity - velocity
        # Get control outputs from PD controllers
        control_ang = self.AngularPDController(error_ang, self.past_error_ang)
        control_dist = self.DistPDController(error_dist, self.past_error_dist)
        control_vel = self.VelocityPDController(error_vel, self.past_error_vel)
        
        # Calculate and return final action
        action = (control_ang + control_dist , control_vel[0], control_vel[1])
        return action

    def returnOnTrack(self, rgb_image):

        ## Cropping image
        strip_height = 20
        middle_height = 65
        top = int(middle_height - strip_height / 2)
        bottom = int(middle_height + strip_height / 2)
        # Crop the strip from the image
        strip = rgb_image[top:bottom, :]
        ## Mask
        hsv = cv2.cvtColor(strip, cv2.COLOR_BGR2HSV)
        mask_green = cv2.inRange(hsv, (36, 25, 25), (70, 255,255))
        #slice the green
        imask_green = mask_green>0
        gray_mask = imask_green.astype(np.uint8)
        gray_mask = gray_mask*255

        strip_height, strip_width = gray_mask.shape
