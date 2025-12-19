import cv2
from ultralytics import YOLO

class PeopleCountingApp:
    def __init__(self, model_path='yolov8n.pt'):
        self.model = YOLO(model_path)
    
    def analyze_frame(self, frame):
        # Run YOLO detection for people (class 0)
        results = self.model.predict(frame, conf=0.5, classes=[0], verbose=False)
        
        # Count people
        people_count = 0
        for r in results:
            boxes = r.boxes
            people_count = len(boxes)
        
        # Optional: draw boxes on frame
        annotated_frame = frame.copy()
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = box.conf[0].item()
                label = f"Person {conf:.2f}"
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(annotated_frame, label, (x1, y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
        text = f"People: {people_count}"
        text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
        text_x = annotated_frame.shape[1] - text_size[0] - 20
        text_y = 50
        cv2.putText(annotated_frame, text, (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        return people_count, annotated_frame
    
    def count_people_in_video(self, video_path, output_path='output.mp4'):
        cap = cv2.VideoCapture(video_path)
        people_timeline = []

        # VideoWriter setup
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        while True:
            success, frame = cap.read()
            if not success:
                break
            
            count, annotated_frame = self.analyze_frame(frame)
            timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
            people_timeline.append({
                'time': timestamp,
                'people': count
            })  
            out.write(annotated_frame)

        cap.release()
        out.release()
        return people_timeline

# Usage
analyzer = PeopleCountingApp()
timeline = analyzer.count_people_in_video('video/street.mp4')

# Print peak people count
print(f"Peak people count: {max(t['people'] for t in timeline)}")
