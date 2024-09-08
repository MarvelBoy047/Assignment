import cv2
from motpy import Detection, MultiObjectTracker
import torch
import streamlit as st
import os
import tempfile
import json
import subprocess

# Function to draw bounding boxes on the frame
def draw_boxes(frame, track_results, id_dict):
    for object in track_results:
        x, y, w, h = object.box
        x, y, w, h = int(x), int(y), int(w), int(h)
        object_id = object.id
        confidence = object.score
        cv2.rectangle(frame, (x, y), (w, h), (0, 255, 0), 2)
        cv2.putText(
            frame,
            f"{str(id_dict[object_id])}: {str(round(confidence, 2))}",
            (x, y - 10),
            cv2.FONT_HERSHEY_PLAIN,
            2,
            (0, 255, 0),
            2,
        )
    cv2.putText(
        frame,
        "People Count: {}".format(len(track_results)),
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2,
    )

# Update ID dictionary with new object IDs
def update_id_dict(id_dict, j, track_results):
    for track_result in track_results:
        if track_result.id not in id_dict:
            id_dict[track_result.id] = j
            j += 1
    return id_dict, j

# Video processing function
def process_video(video_file):
    model = torch.hub.load("ultralytics/yolov5", "yolov5s", pretrained=True)

    # Save the uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
        tmp_file.write(video_file.read())
        video_path = tmp_file.name

    # Read the video and prepare for processing
    cap = cv2.VideoCapture(video_path)
    cap_fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Prepare the output video path
    output_filename = os.path.splitext(video_file.name)[0] + "_annotated.mp4"
    output_path = os.path.join("processed_videos", output_filename)
    os.makedirs("processed_videos", exist_ok=True)

    # Create VideoWriter for output video
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, cap_fps, (width, height))

    # Initialize MultiObjectTracker and ID dictionary
    tracker = MultiObjectTracker(dt=1 / cap_fps, tracker_kwargs={"max_staleness": 10})
    id_dict = {}
    j = 0

    with st.spinner("Processing video..."):
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            results = model(frame)
            output = results.pandas().xyxy[0]

            # Filter objects with label "person"
            objects = output[output["name"] == "person"]
            detections = []

            # Pass YOLO detections to motpy tracker
            for index, obj in objects.iterrows():
                coordinates = [
                    int(obj["xmin"]),
                    int(obj["ymin"]),
                    int(obj["xmax"]),
                    int(obj["ymax"]),
                ]
                detections.append(
                    Detection(
                        box=coordinates,
                        score=obj["confidence"],
                        class_id=obj["class"],
                    )
                )

            # Perform object tracking
            tracker.step(detections=detections)
            track_results = tracker.active_tracks()

            # Update ID dictionary and draw boxes on frame
            id_dict, j = update_id_dict(id_dict, j, track_results)
            draw_boxes(frame, track_results, id_dict)

            # Write the frame to the output video
            out.write(frame)

    cap.release()
    out.release()

    # Remove temporary video file
    os.remove(video_path)

    # Update the log file 
    update_log_file(video_file.name, output_filename, "processed_videos/log.json")

    st.success("Video processing completed successfully!")
    st.balloons()

    return output_path


# Check if the video was already processed and return output file
def get_processed_output_file(video_file, log_file_path):
    try:
        with open(log_file_path, "r") as f:
            log_data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None

    for entry in log_data:
        if entry["original_file"] == video_file.name:
            return entry["output_file"]
    return None

# Update the log file after processing
def update_log_file(original_file, output_file, log_file_path):
    try:
        with open(log_file_path, "r") as f:
            log_data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        log_data = []

    log_data.append({"original_file": original_file, "output_file": output_file})

    with open(log_file_path, "w") as f:
        json.dump(log_data, f, indent=4)

# Re-encode video using FFmpeg for compatibility
def reencode_video(input_path, output_path):
    """Re-encodes the video using FFmpeg to ensure compatibility with Streamlit."""
    command = f"ffmpeg -y -i \"{input_path}\" -vcodec libx264 -acodec aac \"{output_path}\""
    result = subprocess.run(command, shell=True, capture_output=True)

    # Debugging output to check if FFmpeg command runs properly
    if result.returncode != 0:
        st.error(f"Error re-encoding video: {result.stderr.decode()}")
        print(f"FFmpeg Error: {result.stderr.decode()}")
    else:
        print(f"Re-encoding successful: {output_path}")


# Streamlit app entry point
if __name__ == "__main__":
    st.title("Video Annotation App")
    log_file_path = "processed_videos/log.json"
    os.makedirs("processed_videos", exist_ok=True)

    uploaded_file = st.file_uploader("Choose a video file", type=["mp4"])

    if uploaded_file is not None:
        # Check if video has already been processed
        output_filename = get_processed_output_file(uploaded_file, log_file_path)
        
        if output_filename:
            st.info("This video has already been processed.")
            st.snow()

            # Fetch the output file directly from the log
            output_path = os.path.join("processed_videos", output_filename).replace("\\", "/")

            # Provide the reencoded video for download
            with open(output_path, "rb") as f:
                video_bytes = f.read()
                st.download_button(
                    label="Download Annotated Video",
                    data=video_bytes,
                    file_name=output_filename,
                    mime="video/mp4"
                )
        else:
            # Process the video
            processed_video_path = process_video(uploaded_file)

            # Generate the re-encoded output path
            reencoded_output_path = os.path.join("processed_videos", os.path.splitext(uploaded_file.name)[0] + "_reencoded.mp4").replace("\\", "/")

            # Re-encode the video
            reencode_video(processed_video_path, reencoded_output_path)

            # Provide the reencoded video for download
            with open(reencoded_output_path, "rb") as f:
                video_bytes = f.read()
                st.download_button(
                    label="Download Re-encoded Video",
                    data=video_bytes,
                    file_name=reencoded_output_path,
                    mime="video/mp4"
                )
