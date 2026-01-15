"""
Posture Analysis Module
Analyzes body posture, movement, and activities from pose keypoints
"""

import cv2
import numpy as np
from src.core.emotion_detector import SimpleEmotionDetector


class PostureAnalyzer:
    def __init__(self):
        self.movement_history = []
        self.pos_history = []  # For velocity calculation
        self.max_history = 30
        
        # Initialize emotion detector
        print("Loading emotion detection AI model...")
        try:
            self.emotion_detector = SimpleEmotionDetector()
        except Exception as e:
            print(f"⚠ Emotion detection failed: {e}")
            self.emotion_detector = None
        
    def analyze_posture(self, points):
        """Analyze posture from keypoints"""
        if not points[1] or not points[8] or not points[11]:  # Neck, RHip, LHip
            return None
        
        neck = np.array(points[1][:2])
        r_hip = np.array(points[8][:2])
        l_hip = np.array(points[11][:2])
        
        # Calculate spine angle (neck to mid-hips)
        mid_hip = (r_hip + l_hip) / 2
        spine_vector = neck - mid_hip
        vertical = np.array([0, -1])  # Up direction
        
        # Angle from vertical
        angle = np.degrees(np.arctan2(spine_vector[0], -spine_vector[1]))
        
        # Posture classification
        if abs(angle) < 15:
            status = "Excellent"
            color = (0, 255, 0)
        elif abs(angle) < 30:
            status = "Good"
            color = (0, 200, 100)
        elif abs(angle) < 50:
            status = "Fair"
            color = (200, 150, 0)
        else:
            status = "Poor (Bad Posture)"
            color = (0, 0, 255)
        
        # Check shoulder alignment
        if points[2] and points[5]:  # RShoulder, LShoulder
            r_shoulder = np.array(points[2][:2])
            l_shoulder = np.array(points[5][:2])
            shoulder_diff = abs(r_shoulder[1] - l_shoulder[1])
            shoulder_aligned = shoulder_diff < 20
        else:
            shoulder_aligned = None
        
        return {
            'status': status,
            'angle': angle,
            'color': color,
            'shoulder_aligned': shoulder_aligned
        }
    
    def analyze_movement(self, points):
        """Analyze movement energy and velocity from keypoint variance"""
        if not points[1]:
            return {'energy': 'Initializing', 'sentiment': 'N/A', 'movement_score': 0, 'velocity': 0, 'color': (100, 100, 100)}
        
        # Get neck position (relatively stable reference point)
        neck_pos = np.array(points[1][:2])
        
        # Add to history
        self.movement_history.append(neck_pos)
        if len(self.movement_history) > self.max_history:
            self.movement_history.pop(0)
        
        # Calculate velocity
        velocity = 0
        if len(self.movement_history) >= 2:
            velocity = np.linalg.norm(self.movement_history[-1] - self.movement_history[-2])
        
        if len(self.movement_history) < 2:
            return {'energy': 'Initializing', 'sentiment': 'N/A', 'movement_score': 0, 'velocity': velocity, 'color': (100, 100, 100)}
        
        # Calculate movement variance
        positions = np.array(self.movement_history)
        movement = np.var(positions, axis=0).sum()
        
        if movement < 5:
            energy = "Low (Calm/Still)"
            sentiment = "Relaxed/Focused"
            color = (255, 200, 100)
        elif movement < 20:
            energy = "Medium (Active)"
            sentiment = "Engaged/Working"
            color = (100, 255, 100)
        elif movement < 50:
            energy = "High (Moving)"
            sentiment = "Energetic/Excited"
            color = (0, 200, 255)
        else:
            energy = "Very High (Dynamic)"
            sentiment = "Very Active/Restless"
            color = (0, 100, 255)
        
        return {
            'energy': energy,
            'sentiment': sentiment,
            'movement_score': movement,
            'velocity': velocity,
            'color': color
        }
    
    def detect_activity(self, points):
        """Detect specific activities"""
        activities = []
        
        # Check if hands raised (celebrating, waving)
        if points[4] and points[1]:  # RWrist, Neck
            if points[4][1] < points[1][1]:
                activities.append("Right Hand Raised")
        
        if points[7] and points[1]:  # LWrist, Neck
            if points[7][1] < points[1][1]:
                activities.append("Left Hand Raised")
        
        # Check if sitting/standing
        if points[8] and points[10]:  # RHip, RAnkle
            hip_ankle_dist = abs(points[8][1] - points[10][1])
            if hip_ankle_dist < 150:
                activities.append("Sitting")
            else:
                activities.append("Standing")
        
        return activities if activities else ["Normal Pose"]
    
    def analyze_facial_sentiment(self, frame, points):
        """Analyze facial emotions"""
        if self.emotion_detector is None:
            return {
                'emotion': 'N/A',
                'confidence': 0,
                'details': '',
                'color': (200, 200, 200),
                'sentiment': 'Unknown',
                'all_emotions': {}
            }
        
        try:
            result = self.emotion_detector.detect(frame)
            
            if result is None:
                return {
                    'emotion': 'No Face',
                    'confidence': 0,
                    'details': 'No face in frame',
                    'color': (100, 100, 100),
                    'sentiment': 'Unknown',
                    'all_emotions': {}
                }
            
            emotion_name = result['dominant_emotion']
            confidence = result['confidence']
            emotions = result['emotions']
            
            # Map to sentiment and color
            emotion_map = {
                'happy': ('Happy/Joyful', 'Positive', (0, 255, 0)),
                'sad': ('Sad/Down', 'Negative', (255, 0, 100)),
                'angry': ('Angry/Frustrated', 'Negative', (0, 0, 255)),
                'surprise': ('Surprised/Shocked', 'Neutral', (0, 255, 255)),
                'surprised': ('Surprised/Shocked', 'Neutral', (0, 255, 255)),
                'fear': ('Fearful/Scared', 'Negative', (128, 0, 128)),
                'fearful': ('Fearful/Scared', 'Negative', (128, 0, 128)),
                'disgust': ('Disgusted', 'Negative', (0, 128, 128)),
                'disgusted': ('Disgusted', 'Negative', (0, 128, 128)),
                'neutral': ('Neutral/Calm', 'Neutral', (200, 200, 200))
            }
            
            display, sentiment, color = emotion_map.get(emotion_name.lower(), (emotion_name, 'Unknown', (150, 150, 150)))
            
            # Draw face box
            x, y, w, h = result['face_region']
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, f"{display}: {confidence:.0f}%", (x, y-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            # Draw emotion bars
            y_offset = y + h + 10
            for emo, score in sorted(emotions.items(), key=lambda x: x[1], reverse=True):
                if score > 5:
                    bar_width = int(score / 100 * 150)
                    cv2.rectangle(frame, (x, y_offset), (x + bar_width, y_offset + 12), (0, 200, 100), -1)
                    cv2.putText(frame, f"{emo}: {score:.0f}%", (x + 160, y_offset + 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
                    y_offset += 15
            
            details = " | ".join([f"{e}:{v:.0f}%" for e, v in sorted(emotions.items(), key=lambda x: x[1], reverse=True)[:3]])
            
            return {
                'emotion': display,
                'sentiment': sentiment,
                'confidence': int(confidence),
                'details': details,
                'color': color,
                'all_emotions': emotions
            }
            
        except Exception as e:
            return {
                'emotion': 'Error',
                'confidence': 0,
                'details': str(e)[:30],
                'color': (100, 100, 100),
                'sentiment': 'Unknown',
                'all_emotions': {}
            }
