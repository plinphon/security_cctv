import cv2
from ultralytics import YOLO  # assuming YOLOv8

# Load YOLO model
model = YOLO('yolov8n.pt')  # or your trained model

# Custom visualization function
def draw_detections(image, results):
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Get bounding box coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            # Get confidence and class
            conf = box.conf[0].item()
            cls = int(box.cls[0].item())
            label = f"{model.names[cls]} {conf:.2f}"
            
            # Draw bounding box
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw label above the box
            cv2.putText(image, label, (x1, y1-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return image

# Open video or camera
cap = cv2.VideoCapture(0)  # 0 for webcam, or replace with 'video.mp4'

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO detection
    results = model.predict(frame, conf=0.3, verbose=False)
    
    # Draw custom detections
    annotated_frame = draw_detections(frame, results)
    
    # Show frame
    cv2.imshow('YOLO Custom Detection', annotated_frame)
    
    # Exit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
