"""
Real-Time Pose Detection with Backend Integration
Sends pose data to FastAPI backend via WebSocket
"""

import cv2
import os
import asyncio
import websockets
import json

from pose_detector import PoseDetector
from posture_analyzer import PostureAnalyzer
from body_science import BodyScience
from visualization import draw_skeleton, draw_info_panel
from logger import MotionLogger


async def send_frame_data(websocket, frame_data):
    """Send frame analysis data to backend"""
    try:
        await websocket.send(json.dumps(frame_data))
        # Wait for acknowledgment
        response = await asyncio.wait_for(
            websocket.recv(), 
            timeout=1.0
        )
        return json.loads(response)
    except asyncio.TimeoutError:
        print("⚠️ Backend timeout")
        return None
    except Exception as e:
        print(f"❌ Error sending data: {e}")
        return None


async def main():
    # Initialize logger
    logger = MotionLogger()
    
    logger.log("=" * 60)
    logger.log("OpenPose Motion Detection System")
    logger.log("With Backend AI Coaching Integration")
    logger.log("=" * 60)
    
    # Load pose detector
    model_file = "openpose/models/pose/coco/pose_iter_440000.caffemodel"
    config_file = "openpose/models/pose/coco/pose_deploy_linevec.prototxt"
    
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
    
    # Connect to backend WebSocket
    backend_url = "ws://localhost:8000/ws/video-analysis"
    logger.log(f"🔌 Connecting to backend at {backend_url}...")
    
    try:
        async with websockets.connect(backend_url) as websocket:
            logger.log("✅ Connected to backend!")
            logger.log("Real-time Analysis with AI Coaching:")
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
                
                # Body Science calculations
                joints = BodyScience.analyze_joints(points)
                symmetry = BodyScience.analyze_symmetry(points)
                cog_data = BodyScience.analyze_center_of_gravity(points)
                
                # Draw info panel
                frame = draw_info_panel(frame, posture, movement, emotion)
                
                # Prepare data for backend
                frame_data = {
                    "frame_num": frame_count,
                    "timestamp": cv2.getTickCount() / cv2.getTickFrequency(),
                    "keypoints": [
                        list(p) if p is not None else None 
                        for p in points
                    ],
                    "joints": joints,
                    "symmetry": symmetry,
                    "balance": cog_data if cog_data else {},
                    "posture": posture if posture else {},
                    "movement": movement,
                    "emotion": emotion,
                    "activities": activities
                }
                
                # Send to backend (every frame)
                if frame_count % 1 == 0:  # Send every frame
                    response = await send_frame_data(websocket, frame_data)
                    
                    if response and response.get("status") == "received":
                        # Backend acknowledged
                        pass
                
                # Local logging (every 30 frames)
                if frame_count % 30 == 0:
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
                
                # Show frame
                cv2.imshow('Motion & Emotion Analysis (Connected to AI)', frame)
                
                # Check for quit
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
                # Small async yield
                await asyncio.sleep(0.001)
    
    except ConnectionRefusedError:
        logger.log("❌ Backend not running! Start backend first:")
        logger.log("   cd src && uvicorn main:app --reload")
        logger.log("\n⚠️ Running in LOCAL MODE (no AI coaching)")
        
        # Fallback to local mode without backend
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            points, _ = detector.detect(frame)
            frame = draw_skeleton(frame, points, detector.pose_pairs)
            
            posture = postureAnalyzer.analyze_posture(points)
            movement = postureAnalyzer.analyze_movement(points)
            emotion = postureAnalyzer.analyze_facial_sentiment(frame, points)
            
            frame = draw_info_panel(frame, posture, movement, emotion)
            cv2.imshow('Motion Analysis (LOCAL MODE)', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    finally:
        # Clean up
        cap.release()
        cv2.destroyAllWindows()
        logger.close()


if __name__ == "__main__":
    asyncio.run(main())