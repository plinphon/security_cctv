import cv2
import numpy as np

def draw_quadrilateral(frame, points, color=(0,0,255), thickness=2):
    """
    Draws a quadrilateral on the image.
    
    Parameters:
        frame (numpy.ndarray): The image to draw on
        points (list): List of 4 points [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
        color (tuple): BGR color
        thickness (int): Line thickness
    """
    annotated_frame = frame.copy()
    pts = np.array(points, np.int32).reshape((-1,1,2))
    cv2.polylines(annotated_frame, [pts], isClosed=True, color=color, thickness=thickness)
    return annotated_frame

# Example usage
frame = cv2.imread('image.png')
if frame is None:
    print("Image not found!")
else:
    quad_points = [[5,80],[140,40],[500,100],[5,330]]
    annotated_frame = draw_quadrilateral(frame, quad_points)

    cv2.imshow("Quadrilateral Area", annotated_frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
