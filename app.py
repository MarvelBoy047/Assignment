import cv2
from motpy import Detection, MultiObjectTracker
import torch
import streamlit as st
import os
import tempfile
import json
import subprocess
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import google.generativeai as genai
import base64
import time

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
    """Checks if a video has already been processed based on the log file."""
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
    """Updates the log file with a new entry."""
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


def get_emotions_log_data(log_file_path):
    """Reads emotion log data from the log file."""
    try:
        with open(log_file_path, "r") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None

def update_emotions_log_file(log_file_path, video_filename):
    """Updates the emotion log file with a new entry."""
    try:
        with open(log_file_path, "r") as f:
            log_data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        log_data = []

    log_data.append(video_filename)  # Add the filename to the log

    with open(log_file_path, "w") as f:
        json.dump(log_data, f, indent=4)


def perform_emotion_detection_and_annotation(video_path, output_folder, log_file_path):
    webcam = None  # Initialize webcam to None at the beginning
    try:
        emotions_log_data = get_emotions_log_data(log_file_path)
        if emotions_log_data and os.path.basename(video_path) in emotions_log_data:
            st.info("Emotion detection for this video has already been completed.")
            st.snow()
            return None

        model = load_model("facialemotionmodel.h5")
        st.success("Emotion Detection Model loaded successfully!")

        haar_file = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        face_cascade = cv2.CascadeClassifier(haar_file)

        labels = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise'}

        webcam = cv2.VideoCapture(video_path)  # Assign webcam here
        if not webcam.isOpened():
            raise Exception("Failed to open video.")

        frame_width = int(webcam.get(3))
        frame_height = int(webcam.get(4))
        fps = webcam.get(cv2.CAP_PROP_FPS)

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        output_filename = os.path.basename(video_path).replace(".mp4", "_emotions.mp4")
        output_path = os.path.join(output_folder, output_filename)
        os.makedirs(output_folder, exist_ok=True)
        out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

        while True:
            ret, im = webcam.read()
            if not ret:
                st.error("Failed to capture image from video.")
                break

            if im is not None:
                gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, 1.3, 5)

                for (p, q, r, s) in faces:
                    image = gray[q:q+s, p:p+r]
                    cv2.rectangle(im, (p, q), (p+r, q+s), (255, 0, 0), 2)
                    image = cv2.resize(image, (48, 48))

                    img = np.array(image).reshape(1, 48, 48, 1)
                    pred = model.predict(img / 255.0)
                    prediction_label = labels[pred.argmax()]
                    cv2.putText(im, '% s' % (prediction_label), (p-10, q-10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 0, 255))

                out.write(im)

        update_emotions_log_file(log_file_path, os.path.basename(video_path))

        st.success("Emotion detection and annotation complete!")
        return output_path

    except Exception as e:
        st.error(f"Error accessing video: {str(e)}")
    
    finally:
        if webcam is not None:  # Check if webcam was initialized before releasing
            webcam.release()
        cv2.destroyAllWindows()


def verify_gemini_key(api_key):
  """Verifies the Gemini API key by sending a test request."""
  os.environ["GEMINI_API_KEY"] = api_key
  try:
      genai.configure(api_key=api_key)
      model = genai.GenerativeModel(model_name="gemini-1.5-flash-exp-0827")
      chat_session = model.start_chat()
      response = chat_session.send_message("Hello, Gemini!")
      if "Hello!" in response.text:
          return True
      else:
          return False
  except Exception as e:
      st.error(f"Error verifying Gemini API key: {str(e)}")
      return False


# Streamlit app entry point
if __name__ == "__main__":
    st.title("Video Annotation and Emotion Detection App")
    log_file_path = "processed_videos/log.json"
    emotions_log_file_path = "processed_emotions/log.json"
    os.makedirs("processed_videos", exist_ok=True)
    os.makedirs("processed_emotions", exist_ok=True)

    # --- UPLOAD VIDEO ---
    uploaded_file = st.file_uploader("Choose a video file", type=["mp4"])

    # --- PROCESS OR REUSE VIDEO ---
    if uploaded_file is not None:
        # Check if video has already been processed
        output_filename = get_processed_output_file(uploaded_file, log_file_path)

        if output_filename:
            st.info("This video has already been processed.")
            st.snow()

            # Fetch the output file directly from the log
            output_path = os.path.join("processed_videos", output_filename)

            # Provide the reencoded video for download
            with open(output_path, "rb") as f:
                video_bytes = f.read()
                st.download_button(
                    label="Download Annotated Video",
                    data=video_bytes,
                    file_name=output_filename,
                    mime="video/mp4"
                )

            # Perform emotion detection (after original processing is done)
            emotions_output_path = perform_emotion_detection_and_annotation(
                output_path, "processed_emotions", emotions_log_file_path
            )

            if emotions_output_path:
                with open(emotions_output_path, "rb") as f:
                    video_bytes = f.read()
                    st.download_button(
                        label="Download Emotion Annotated Video",
                        data=video_bytes,
                        file_name=os.path.basename(emotions_output_path),
                        mime="video/mp4"
                    )

        else:
            # Process the video (with object tracking)
            processed_video_path = process_video(uploaded_file)

            # Perform emotion detection (after original processing is done)
            emotions_output_path = perform_emotion_detection_and_annotation(
                processed_video_path, "processed_emotions", emotions_log_file_path
            )

            if emotions_output_path:
                with open(emotions_output_path, "rb") as f:
                    video_bytes = f.read()
                    st.download_button(
                        label="Download Emotion Annotated Video",
                        data=video_bytes,
                        file_name=os.path.basename(emotions_output_path),
                        mime="video/mp4"
                    )


        # --- GEMINI API KEY INPUT AND VERIFICATION ---
        gemini_api_key = st.text_input("Enter your Gemini API key:")


        # --- PASTE CLIPBOARD BUTTON (FOR CONVENIENCE) ---
        if st.button("Paste Clipboard"):
            try:
                with open("clipboard.txt", "w") as f:
                    clipboard_data = st.get_clipboard()
                    f.write(clipboard_data)
                st.success("Clipboard data pasted to file!")
            except Exception as e:
                st.error(f"Error pasting clipboard: {e}")


        # --- VERIFY KEY BUTTON ---
        if st.button("Verify API Key"):
            if verify_gemini_key(gemini_api_key):
                st.success("Gemini API key verified successfully!")
            else:
                st.error("Gemini API key verification failed. Please check the key and try again.")


        # --- CHAT WITH GEMINI (IF KEY IS VALID) ---
        if gemini_api_key and verify_gemini_key(gemini_api_key):
            user_question = st.text_area("Ask a question about the video:")
            if user_question:
                genai.configure(api_key=gemini_api_key)
                model = genai.GenerativeModel(model_name="models/gemini-1.5-flash")

                # ---  OPTION TO RE-UPLOAD THE VIDEO HERE (NEW PART) ---
                reupload_video = st.file_uploader("Re-upload the video for Gemini", type=["mp4"])
                if reupload_video is not None:
                    video_file = genai.upload_file(path=reupload_video, mime_type="video/mp4")  

                    # Wait for processing (add spinner etc.)
                    while video_file.state.name == "PROCESSING":
                        st.write("Processing video...")
                        time.sleep(5)
                        video_file = genai.get_file(video_file.name)

                    if video_file.state.name == "FAILED":
                        st.error("Video processing failed!")
                    else:
                        # Construct Prompt and send to Gemini
                        prompt = f"Based on this video, {user_question}" 
                        response = model.generate_content([prompt, video_file], request_options={"timeout": 600})
                        st.write(response.text) 
                        
                else: # If no reupload, use the original video path
                     # (Assume processed_video_path is from your earlier processing)
                     with open(processed_video_path, "rb") as f:
                         video_file = genai.upload_file(path=processed_video_path, mime_type="video/mp4")

                     # Wait for processing (add spinner etc.)
                     while video_file.state.name == "PROCESSING":
                        st.write("Processing video...")
                        time.sleep(5)
                        video_file = genai.get_file(video_file.name)

                     if video_file.state.name == "FAILED":
                        st.error("Video processing failed!")
                     else:
                        # Construct Prompt and send to Gemini
                        prompt = f"Based on this video, {user_question}" 
                        response = model.generate_content([prompt, video_file], request_options={"timeout": 600})
                        st.write(response.text)