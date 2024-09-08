```markdown
here are all required files ---> https://drive.google.com/drive/folders/1nI1fMD5rWtoQPiVBApUhSyCcNggnLbV2?usp=sharing
```

# Video Annotation, Emotion Detection, and Gemini Integration App

This Streamlit application allows users to upload a video file, process it with object tracking and emotion detection, and then interact with the video content using Gemini, Google's powerful large language model.

## Features

*   **Object Tracking:** Uses YOLOv5 for object detection and MultiObjectTracker (MOTPy) for tracking people in the video. 
*   **Emotion Detection:** Leverages a pre-trained facial emotion recognition model to identify emotions (angry, disgust, fear, happy, neutral, sad, surprise) in the video.
*   **Gemini Integration:** Enables users to ask questions about the video's content using Gemini.
*   **Re-upload Option:** Allows users to re-upload a different video for each question they ask Gemini.
*   **Downloadable Outputs:**  Provides options to download annotated videos (with bounding boxes and emotion labels).
*   **Log File:** Keeps track of processed videos and their corresponding outputs.


## How it Works

1.  **Upload Video:** The user selects a video file (MP4 format) to upload using the file uploader.
2.  **Video Processing (Object Tracking):** 
    *   The application uses YOLOv5 to detect people in each frame of the video.
    *   The detected objects are then tracked using the MultiObjectTracker library. 
    *   Bounding boxes and object IDs are drawn on the frames.
    *   The annotated video is saved in the `processed_videos` folder.
3.  **Emotion Detection:**
    *   The processed video is then analyzed using a pre-trained facial emotion recognition model.
    *   Emotion labels are overlaid on detected faces in the video.
    *   The emotion-annotated video is saved in the `processed_emotions` folder.
4.  **Gemini Interaction:**
    *   The user enters their Gemini API key. 
    *   The user can then type a question about the video.
    *   The user can choose to re-upload a different video if needed.
    *   The application uploads the video (either the original or the re-uploaded one) to Gemini. 
    *   Gemini processes the video and generates a response based on the user's question and the video's content.
    *   The response is displayed in the Streamlit UI.
5.  **Download:**  The user can download the processed video files with object tracking and emotion detection information.


## Dependencies

*   Streamlit
*   OpenCV
*   Motpy
*   Torch
*   Ultralytics
*   TensorFlow
*   Google Generative AI

**To install dependencies:**
**How to use `requirements.txt`:**

1.  Save this content as `requirements.txt` in the same directory as your `app.py` file.
2.  When deploying or sharing your project, include `requirements.txt`.
3.  To install all the required packages, run:
    ```bash
    pip install -r requirements.txt
    ```
```bash
pip install streamlit google-generativeai opencv-python motpy torch ultralytics tensorflow
```
## Usage

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/MarvelBoy047/Assignment.git
    ```
2.  **Obtain a Gemini API Key:** Get a Gemini API key from the Google Cloud Console.
3.  **Run the app:**
    ```bash
    streamlit run app.py 
    ```
4.  **Enter API Key:** In the Streamlit app, enter your Gemini API key and click the "Verify API Key" button.
5.  **Upload a Video:** Use the file uploader to select a video file (MP4 format).
6.  **Ask Questions:** Once the video is processed, you can ask Gemini questions about its content.


## Folder Structure

```
emotions_detection/
├── app.py             # Main Streamlit application
├── processed_videos/  # Folder to store processed videos with object tracking
├── processed_emotions/ # Folder to store processed videos with emotion detection
└── facialemotionmodel.h5 # Pre-trained facial emotion recognition model
```


## Notes

*   The `facialemotionmodel.h5` file needs to be in the same directory as the `app.py` file.
*   The application uses temporary files during video processing.
*   Large videos may result in longer processing times.
*   The effectiveness of Gemini's responses may vary depending on the video content and the complexity of the question.

```
## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues. 


```
