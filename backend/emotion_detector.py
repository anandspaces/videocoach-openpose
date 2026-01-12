"""
Emotion Detection Module
Uses Haar cascades and facial analysis to detect emotions
"""

import cv2
import numpy as np


class SimpleEmotionDetector:
    """
    Lightweight emotion detector using Haar cascades + analysis
    Detects: happy, sad, angry, neutral, surprised, fearful, disgusted
    """
    
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        self.face_alt = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_alt.xml'
        )
        self.face_alt2 = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml'
        )
        self.smile_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_smile.xml'
        )
        self.eye_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_eye.xml'
        )
        print("✓ Emotion Detector loaded (multi-cascade face + smile + eye analysis)")
    
    def detect(self, frame):
        """Detect emotion in frame with multiple fallback strategies"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Enhance contrast for better detection
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
        
        # Try multiple cascade classifiers with different parameters
        faces = self._detect_faces_robust(gray)
        
        if len(faces) == 0:
            return None
        
        # Process largest face
        (x, y, w, h) = max(faces, key=lambda f: f[2] * f[3])
        
        # Expand face region slightly for better emotion detection
        padding = int(w * 0.1)
        x = max(0, x - padding)
        y = max(0, y - padding)
        w = min(gray.shape[1] - x, w + padding * 2)
        h = min(gray.shape[0] - y, h + padding * 2)
        
        face_roi_gray = gray[y:y+h, x:x+w]
        face_roi_color = frame[y:y+h, x:x+w]
        
        # Detect smiles in face with relaxed parameters
        smiles = self.smile_cascade.detectMultiScale(
            face_roi_gray, 
            scaleFactor=1.1, 
            minNeighbors=10,  # Lower threshold for better detection
            minSize=(int(w*0.15), int(h*0.15))
        )
        
        # Detect eyes in face  
        eyes = self.eye_cascade.detectMultiScale(
            face_roi_gray, 
            scaleFactor=1.05, 
            minNeighbors=3,
            minSize=(int(w*0.1), int(h*0.1))
        )
        
        # Analyze face texture and brightness
        brightness = cv2.mean(face_roi_gray)[0]
        # Variance indicates emotion intensity
        variance = cv2.Laplacian(face_roi_gray, cv2.CV_64F).var()
        
        # Analyze mouth region for smile detection
        mouth_smile_score = self._detect_mouth(face_roi_gray, h)
        
        # Calculate emotion scores based on features
        emotions = self._calculate_emotions(
            smiles, eyes, brightness, variance, 
            face_roi_gray, mouth_smile_score
        )
        
        return {
            'face_region': (x, y, w, h),
            'dominant_emotion': max(emotions, key=emotions.get),
            'confidence': max(emotions.values()),
            'emotions': emotions
        }
    
    def _detect_faces_robust(self, gray):
        """Try multiple cascade classifiers to detect faces"""
        faces = []
        
        # Method 1: Default cascade with sensitive parameters
        f1 = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.05, minNeighbors=4, minSize=(30, 30)
        )
        faces.extend(f1)
        
        # Method 2: Alternative cascade
        if len(faces) == 0:
            f2 = self.face_alt.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
            )
            faces.extend(f2)
        
        # Method 3: Another alternative
        if len(faces) == 0:
            f3 = self.face_alt2.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=6, minSize=(30, 30)
            )
            faces.extend(f3)
        
        # Method 4: Very sensitive search if still no faces
        if len(faces) == 0:
            f4 = self.face_cascade.detectMultiScale(
                gray, scaleFactor=1.03, minNeighbors=2, minSize=(20, 20)
            )
            faces.extend(f4)
        
        # Remove duplicates (overlapping detections)
        if faces:
            faces = self._remove_overlaps(faces)
        
        return faces
    
    def _remove_overlaps(self, faces):
        """Remove duplicate/overlapping face detections"""
        if len(faces) <= 1:
            return faces
        
        # Sort by area (largest first)
        faces = sorted(faces, key=lambda f: f[2]*f[3], reverse=True)
        
        result = [faces[0]]
        for face in faces[1:]:
            x1, y1, w1, h1 = face
            overlaps = False
            for rx, ry, rw, rh in result:
                # Check if faces overlap
                if not (x1 + w1 < rx or rx + rw < x1 or 
                        y1 + h1 < ry or ry + rh < y1):
                    overlaps = True
                    break
            if not overlaps:
                result.append(face)
        
        return result
    
    def _detect_mouth(self, face_roi_gray, face_height):
        """Detect smile by analyzing mouth region"""
        try:
            # Mouth is in lower 1/3 of face
            mouth_y = int(face_height * 0.6)
            mouth_roi = face_roi_gray[mouth_y:, :]
            
            # High variance in mouth region indicates smiling
            mouth_var = cv2.Laplacian(mouth_roi, cv2.CV_64F).var()
            
            # Normalize to 0-100 scale
            mouth_score = min(100, mouth_var / 5)
            return mouth_score
        except:
            return 0
    
    def _calculate_emotions(self, smiles, eyes, brightness, variance, face_roi, mouth_smile):
        """Calculate emotion probabilities"""
        smile_count = len(smiles)
        eye_count = len(eyes)
        
        # Enhanced emotion scores with mouth analysis
        emotions = {
            'happy': 15 + (smile_count * 35) + (eye_count * 8) + (mouth_smile * 0.3),
            'neutral': 35 - (smile_count * 8) - (eye_count * 3) + (50 - mouth_smile) * 0.2,
            'sad': 10 - (smile_count * 12) + (50 - mouth_smile) * 0.1,
            'angry': 12 + (max(0, 100 - brightness) / 4),
            'surprised': max(0, eye_count * 30 - 15),
            'fearful': max(0, (100 - brightness) / 2.5),
            'disgusted': 8 + (max(0, brightness - 140) / 2)
        }
        
        # Normalize to percentages
        total = sum(emotions.values())
        if total > 0:
            emotions = {k: (v / total) * 100 for k, v in emotions.items()}
        
        return emotions
