import cv2
from ultralytics import YOLO

car_ids = set()

model = YOLO("yolov8n.pt")
camera = cv2.VideoCapture("video.mp4")

while camera.isOpened():
    success, frame = camera.read()

    if not success:
        break

    result = model.track(frame, persist=True, verbose=False)[0]

    for box in result.boxes:
        class_id = int(box.cls.item())
        object_name = result.names[class_id]

        if object_name == "car" and box.id is not None:
            tracking_id = int(box.id.item())
            car_ids.add(tracking_id)

    cv2.imshow("Car tracking", result.plot())

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()

print(f"Total tracked cars: {len(car_ids)}")