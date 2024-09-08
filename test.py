import cv2
import os

def analyze_video(video_file_path):
    """Analyzes a video file and provides information about potential issues."""

    print(f"Analyzing video: {video_file_path}")

    try:
        cap = cv2.VideoCapture(video_file_path)
        if not cap.isOpened():
            print(f"Error opening video file: {video_file_path}")
            return

        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))

        # Get codec name correctly
        codec_name = cv2.VideoWriter_fourcc(*"mp4v")  # Or use the appropriate codec

        print(f"Video properties:")
        print(f"  - FPS: {fps}")
        print(f"  - Width: {width}")
        print(f"  - Height: {height}")
        print(f"  - Codec: {codec_name}")

        # Check for invalid frames
        invalid_frames = 0
        frame_count = 0
        while True:
            ret, frame = cap.read()
            frame_count += 1
            if not ret:
                break
            if frame is None:
                invalid_frames += 1

        cap.release()

        print(f"  - Frame count: {frame_count}")
        print(f"  - Invalid frames: {invalid_frames}")

        if invalid_frames > 0:
            print("WARNING: Found invalid frames in the video. This could indicate corruption.")

    except Exception as e:
        print(f"Error analyzing video: {e}")



if __name__ == "__main__":
    video_file_path = "processed_videos\Matching_480p_annotated.mp4"
    if os.path.exists(video_file_path):
        analyze_video(video_file_path)
    else:
        print("Error: Video file not found.")