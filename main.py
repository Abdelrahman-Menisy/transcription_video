from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from typing import Optional
import whisper
import os
import shutil
import cv2
from moviepy.editor import ImageSequenceClip, AudioFileClip, VideoFileClip
from tqdm import tqdm

app = FastAPI()

FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.8
FONT_THICKNESS = 2

class MediaTranscriber:
    def __init__(self, model_path, input_path):
        # Load the model from a local path
        self.model = whisper.load_model(model_path)
        self.input_path = input_path
        self.is_video = input_path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))
        self.audio_path = input_path if not self.is_video else ''
        self.text_array = []
        self.fps = 30  # Default FPS, will be updated for video files
        self.char_width = 10  # Default character width, will be updated later

    def transcribe_media(self):
        result = self.model.transcribe(self.audio_path or self.input_path)
        
        if self.is_video:
            cap = cv2.VideoCapture(self.input_path)
            self.fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            asp = 16/9
            ret, frame = cap.read()
            if ret:
                frame = frame[:, int(int(width - 1 / asp * height) / 2):width - int((width - 1 / asp * height) / 2)]
                width = frame.shape[1] - (frame.shape[1] * 0.1)
            cap.release()
        else:
            width = 1280  # Default width for audio files

        for segment in tqdm(result["segments"]):
            text = segment["text"]
            end = segment["end"]
            start = segment["start"]
            total_frames = int((end - start) * self.fps)
            start_frame = int(start * self.fps)
            words = text.split()
            
            current_line = []
            current_line_length = 0
            for word in words:
                word_length = len(word) * self.char_width
                if current_line_length + word_length > width:
                    line_text = " ".join(current_line)
                    line_frames = int(len(line_text) / len(text) * total_frames)
                    self.text_array.append([line_text, start_frame, start_frame + line_frames])
                    start_frame += line_frames
                    current_line = [word]
                    current_line_length = word_length
                else:
                    current_line.append(word)
                    current_line_length += word_length + self.char_width  # Add space

            if current_line:
                line_text = " ".join(current_line)
                self.text_array.append([line_text, start_frame, int(end * self.fps)])

    def extract_audio(self):
        if not self.is_video:
            return
        
        self.audio_path = os.path.splitext(self.input_path)[0] + "_audio.mp3"
        video = VideoFileClip(self.input_path)
        audio = video.audio 
        audio.write_audiofile(self.audio_path)
        video.close()

    def create_subtitled_video(self, output_path):
        if not self.is_video:
            return

        temp_dir = "temp_frames"
        os.makedirs(temp_dir, exist_ok=True)

        cap = cv2.VideoCapture(self.input_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        for frame_num in tqdm(range(total_frames)):
            ret, frame = cap.read()
            if not ret:
                break

            for text, start, end in self.text_array:
                if start <= frame_num <= end:
                    text_size, _ = cv2.getTextSize(text, FONT, FONT_SCALE, FONT_THICKNESS)
                    text_x = int((width - text_size[0]) / 2)
                    text_y = height - 50
                    cv2.putText(frame, text, (text_x, text_y), FONT, FONT_SCALE, (255, 255, 255), FONT_THICKNESS, cv2.LINE_AA)
                    break

            cv2.imwrite(os.path.join(temp_dir, f"{frame_num:04d}.jpg"), frame)

        cap.release()

        images = [img for img in os.listdir(temp_dir) if img.endswith(".jpg")]
        images.sort(key=lambda x: int(x.split(".")[0]))

        clip = ImageSequenceClip([os.path.join(temp_dir, image) for image in images], fps=self.fps)
        audio = AudioFileClip(self.audio_path or self.input_path)
        final_clip = clip.set_audio(audio)
        final_clip.write_videofile(output_path)

        shutil.rmtree(temp_dir)
        if self.audio_path and os.path.exists(self.audio_path):
            os.remove(self.audio_path)

@app.post("/transcribe")
async def transcribe_media(file: UploadFile = File(...)):
    model_path = "small"
    input_path = f"temp_{file.filename}"
    output_path = f"subtitled_{file.filename}"

    with open(input_path, "wb") as f:
        f.write(await file.read())

    transcriber = MediaTranscriber(model_path, input_path)

    if transcriber.is_video:
        transcriber.extract_audio()

    transcriber.transcribe_media()

    if transcriber.is_video:
        transcriber.create_subtitled_video(output_path)
        # Return the video file
        return FileResponse(output_path, media_type='video/mp4', headers={"Content-Disposition": f"attachment; filename={output_path}"})
    else:
        # Clean up
        os.remove(input_path)
        return JSONResponse(content={"message": "Audio processed, no video output for audio-only input."})

    # Clean up
    os.remove(input_path)

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)
