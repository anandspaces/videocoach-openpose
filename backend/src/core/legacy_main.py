"""
Real-Time Pose Detection with Emotion Analysis
Uses OpenCV DNN + Simple CNN-based emotion detector

Main application file that orchestrates all modules.
"""

import cv2
import os

from src.core.pose_detector import PoseDetector
from src.core.posture_analyzer import PostureAnalyzer
from src.core.body_science import BodyScience
from src.core.visualization import draw_skeleton, draw_info_panel
from src.core.logger import MotionLogger


def main():
    # Initialize logger
    logger = MotionLogger()
    
    logger.log("=" * 60)
    logger.log("OpenPose Motion Detection System")
    logger.log("With Posture & Emotion Analysis")
    logger.log("=" * 60)
    
    # Load pose detector
    model_file = "src/openpose/models/pose/coco/pose_iter_440000.caffemodel"
    config_file = "src/openpose/models/pose/coco/pose_deploy_linevec.prototxt"
    
    if not os.path.exists(model_file):
        logger.log(f"Error: Model not found at {model_file}")
        logger.close()
        return
    
    logger.log(f"Loading model from: {model_file}")
    detector = PoseDetector(model_file, config_file, use_cuda=False)
    
    # Initialize analyzers
    postureAnalyzer = PostureAnalyzer()
    
    # Open webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        logger.log("Error: Cannot open webcam")
        logger.close()
        return
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    logger.log("Running on CPU")
    logger.log("[OK] Webcam active! Press 'q' to quit")
    logger.log("Real-time Analysis:")
    logger.log("-" * 60)
    
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Detect pose
        points, _ = detector.detect(frame)
        
        # Draw skeleton
        frame = draw_skeleton(frame, points, detector.pose_pairs)
        
        # Analyze posture
        posture = postureAnalyzer.analyze_posture(points)
        movement = postureAnalyzer.analyze_movement(points)
        activities = postureAnalyzer.detect_activity(points)
        emotion = postureAnalyzer.analyze_facial_sentiment(frame, points)
        
        # Draw info panel
        frame = draw_info_panel(frame, posture, movement, emotion)
        
        # Show frame
        cv2.imshow('Motion & Emotion Analysis', frame)
        
        # Terminal output (every frame for detailed logging)
        if frame_count % 1 == 0:
            # Body Science calculations
            joints = BodyScience.analyze_joints(points)
            symmetry = BodyScience.analyze_symmetry(points)
            cog_data = BodyScience.analyze_center_of_gravity(points)
            
            # Log comprehensive frame analysis
            logger.log_frame_analysis(
                frame_count, 
                points, 
                detector.points_names,
                joints,
                symmetry,
                cog_data,
                posture,
                movement,
                emotion,
                activities
            )
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Clean up
    cap.release()
    cv2.destroyAllWindows()
    logger.close()


if __name__ == "__main__":
    main()
