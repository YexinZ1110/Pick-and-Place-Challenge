import cv2
import apriltag
from core.interfaces import ObjectDetector

def read_tag(rbg_img, depth_img, camera_intrinsics):
    gray_image = cv2.cvtColor(rbg_img, cv2.COLOR_BGR2GRAY)
    april_tag_detector = apriltag.Detector()
    april_tag_gray = april_tag_detector.detect(gray_image)
    
    # Extract poses
    poses = []
    for detection in april_tag_gray:
        depth = depth_img[detection.center[1], detection.center[0]]
        position = calculate_position(camera_intrinsics, detection, depth)
        orientation = calculate_orientation(detection)
        poses.append({'ID': detection.tag_id, 'position': position, 'orientation': orientation})
    return poses
        
def calculate_position(camera_intrinsics, detection, depth):
    fx, fy, cx, cy = camera_intrinsics
    x = (detection.center[0] - cx) * depth / fx
    y = (detection.center[1] - cy) * depth / fy
    return [x, y, depth]

def calculate_orientation(detection):
    ############### how to get R? ###############
    R = 
    
    S = 