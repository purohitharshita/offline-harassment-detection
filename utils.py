import librosa
import numpy as np
import cv2

def extract_audio_features(file_path):
    try:
        # Extract MFCCs with shape (40, 130)
        audio, sr = librosa.load(file_path, sr=None)
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
        # Pad or truncate to 130 frames
        if mfcc.shape[1] < 130:
            pad_width = 130 - mfcc.shape[1]
            mfcc = np.pad(mfcc, ((0,0),(0,pad_width)), mode='constant')
        else:
            mfcc = mfcc[:, :130]
        # Add channel dimension
        mfcc = np.expand_dims(mfcc, axis=-1)  # shape (40, 130, 1)
        return mfcc
    except Exception as e:
        print("Audio Feature Extraction Error:", e)
        return None

def extract_video_features(video_path, max_frames=60):
    try:
        cap = cv2.VideoCapture(video_path)
        frames = []
        count = 0
        while count < max_frames and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            # Resize to 250x250 (or the size you used in training, adjust if needed)
            frame = cv2.resize(frame, (250, 250))
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            flat = gray.flatten()
            # Pad/truncate to 62720 per frame
            if len(flat) < 62720:
                flat = np.pad(flat, (0, 62720 - len(flat)), mode='constant')
            else:
                flat = flat[:62720]
            frames.append(flat)
            count += 1
        cap.release()
        # Pad with zeros if less than max_frames
        if len(frames) < max_frames:
            frames += [np.zeros(62720)] * (max_frames - len(frames))
        else:
            frames = frames[:max_frames]
        features = np.array(frames)  # shape (60, 62720)
        print("Extracted features for video:", video_path, "Shape:", features.shape, "Mean:", np.mean(features))
        return features
    except Exception as e:
        print("Video Feature Extraction Error:", e)
        return np.zeros((max_frames, 62720))
