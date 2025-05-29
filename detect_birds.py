from ultralytics import YOLO
import cv2
import os

video_path = "Norwich6.MOV"
print(f"using video file: {video_path}")
output_video_path = "Norwich6_OUTPUT.MOV"
clip_folder = "bird_clips"
clip_duration_seconds = 3
model_path = "yolov8n.pt"  


os.makedirs(clip_folder, exist_ok=True)


model = YOLO(model_path)


cap = cv2.VideoCapture(video_path)
fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30 
width, height = 640, 360  
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(output_video_path, fourcc, fps // 2, (width, height))

frame_num = 0
bird_frames = []

print("\n🚀 Starting detection...")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.rotate(frame, cv2.ROTATE_180)

    
    if frame_num % 2 != 0:
        frame_num += 1
        continue

    frame = cv2.resize(frame, (width, height))
    results = model(frame, verbose=False)[0]

    found_bird = False
    for i in range(len(results.boxes.cls)):
        if int(results.boxes.cls[i]) == 14:  
            found_bird = True
            x1, y1, x2, y2 = map(int, results.boxes.xyxy[i])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, "Bird", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    if found_bird:
        bird_frames.append(frame_num)
        print(f"🕊️  Bird detected at frame {frame_num}")

    if frame_num % 500 == 0:
        print(f"🔄 Processed {frame_num} frames")

    out.write(frame)
    frame_num += 1

cap.release()
out.release()
print(f"\n✅ Detection video saved: {output_video_path}")


def extract_clip(start_frame, end_frame, clip_id):
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    clip_path = os.path.join(clip_folder, f"clip_{clip_id:03d}.mp4")
    out_clip = cv2.VideoWriter(clip_path, fourcc, fps, (width, height))

    bird_found = False

    for _ in range(end_frame - start_frame + 1):
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.rotate(frame, cv2.ROTATE_180)

        frame = cv2.resize(frame, (width, height))
        results = model(frame, verbose=False)[0]

        for i in range(len(results.boxes.cls)):
            if int(results.boxes.cls[i]) == 14:
                bird_found = True
                x1, y1, x2, y2 = map(int, results.boxes.xyxy[i])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, "Bird", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        out_clip.write(frame)

    cap.release()
    out_clip.release()

    if not bird_found:
        os.remove(clip_path)
        print(f"❌ Deleted clip {clip_id:03d}: No bird found.")
    else:
        print(f"🎬 Saved bird clip: {clip_path}")

clip_frame_range = int((clip_duration_seconds * fps) // 2)
bird_frames.sort()
merged_clips = []
last_end = -1

for f in bird_frames:
    start = max(f - clip_frame_range, 0)
    end = f + clip_frame_range
    if start <= last_end:
        continue  
    merged_clips.append((start, end))
    last_end = end

print(f"\n🎞️ Extracting {len(merged_clips)} clips...")

for i, (start, end) in enumerate(merged_clips):
    extract_clip(start, end, i + 1)

print(f"\n✅ Total final clips: {len(merged_clips)} saved in '{clip_folder}'")
