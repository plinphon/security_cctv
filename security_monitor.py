from datetime import datetime
import json
import cv2
from ultralytics import YOLO
import numpy as np

class SecurityMonitor:
    def __init__(self, model_path='yolov8n.pt'):
        self.model = YOLO(model_path)
        self.alert_log = []
    
    def detect_intrusion(self, frame, restricted_area_coordinates):
        results = self.model.predict(frame, conf=0.6, classes=[0], verbose=False)
        alerts = []

        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                
                # Check if person is in restricted area
                pts = np.array(restricted_area_coordinates, np.int32).reshape((-1,1,2))
                if cv2.pointPolygonTest(pts, (center_x, center_y), False) >= 0:
                    alert = {
                        'timestamp': datetime.now().isoformat(),
                        'type': 'intrusion',
                        'location': [float(center_x), float(center_y)],
                        'confidence': float(box.conf[0].item())
                    }
                    alerts.append(alert)
                    self.alert_log.append(alert)
        return alerts
    
    def draw_quadrilateral(self, frame, points, color=(0,0,255), thickness=2):
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

    
    def monitor_video(self, video_path, output_path='security_output.mp4', restricted_area_coordinates=[100,100,400,400]):
        cap = cv2.VideoCapture(video_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        while True:
            success, frame = cap.read()
            if not success:
                break
            
            # Check for intrusions
            alerts = self.detect_intrusion(frame, restricted_area_coordinates)
            
            # Draw restricted area
            frame = self.draw_quadrilateral(frame=frame, points=restricted_area_coordinates)
            
            # Draw bounding boxes and alerts
            for alert in alerts:
                cx, cy = map(int, alert['location'])
                cv2.circle(frame, (cx, cy), 10, (0, 0, 255), -1)
                cv2.putText(frame, "INTRUSION", (cx+15, cy),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
                
            cv2.imshow("Security Monitor", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
            # Write frame to output video
            out.write(frame)
        
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        
        print(f"Finished processing. Output saved to {output_path}")
        # Save alert log to JSON
        with open('alert_log.json', 'w') as f:
            json.dump(self.alert_log, f, indent=2)
        print("Alert log saved to alert_log.json")


# Usage
monitor = SecurityMonitor()
monitor.monitor_video('video/Stealing013_x264.mov', 'security_output.mp4', restricted_area_coordinates=[[5,80],[140,40],[600,100],[5,330]])
