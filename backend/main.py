"""
Headless AI Video Coach System - WITH MEETING SUPPORT
Server-side backend for React frontend - No GUI
Includes video meeting session creation and management
"""

import cv2
import os
import asyncio
import json
import base64
import numpy as np
import logging
import uuid
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from typing import Dict, Any, Optional
import uvicorn

# Pose detection imports
from pose_detector import PoseDetector
from posture_analyzer import PostureAnalyzer
from body_science import BodyScience

# Backend service imports
from services.coach_engine import CoachEngine
from services.state_manager import SessionManager
from websocket.gemini_ws import GeminiClient
from websocket.tts_ws import TTSClient

# Meeting management
from services.meet_session import VideoMeetManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global managers
session_manager = SessionManager()
video_meet_manager = None
gemini_client = None
tts_client = None
pose_detector = None
posture_analyzer = None


def convert_to_serializable(obj):
    """Convert numpy types to Python native types for JSON serialization"""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_to_serializable(item) for item in obj)
    return obj


def decode_base64_frame(base64_str: str) -> np.ndarray:
    """Decode base64 image string to OpenCV frame"""
    try:
        img_data = base64.b64decode(base64_str)
        nparr = np.frombuffer(img_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return frame
    except Exception as e:
        logger.error(f"Failed to decode frame: {e}")
        return None


def process_frame(frame: np.ndarray, frame_count: int) -> Dict[str, Any]:
    """Process a single frame and return analysis data"""
    # Detect pose
    points, _ = pose_detector.detect(frame)
    
    # Analyze posture
    posture = posture_analyzer.analyze_posture(points)
    movement = posture_analyzer.analyze_movement(points)
    activities = posture_analyzer.detect_activity(points)
    emotion = posture_analyzer.analyze_facial_sentiment(frame, points)
    
    # Body Science calculations
    joints = BodyScience.analyze_joints(points)
    symmetry = BodyScience.analyze_symmetry(points)
    cog_data = BodyScience.analyze_center_of_gravity(points)
    
    # Prepare analysis data
    frame_data = {
        "frame_num": int(frame_count),
        "timestamp": float(cv2.getTickCount() / cv2.getTickFrequency()),
        "keypoints": [
            {"x": float(p[0]), "y": float(p[1]), "confidence": float(p[2])} if p is not None else None 
            for p in points
        ],
        "joints": {k: float(v) for k, v in joints.items()},
        "symmetry": {k: float(v) for k, v in symmetry.items()},
        "balance": {
            "cog": [float(cog_data['cog'][0]), float(cog_data['cog'][1])],
            "balance_score": float(cog_data['balance_score'])
        } if cog_data else {},
        "posture": {
            "status": posture['status'],
            "angle": float(posture['angle']),
            "shoulder_aligned": bool(posture['shoulder_aligned']) if posture['shoulder_aligned'] is not None else None
        } if posture else {},
        "movement": {
            "energy": movement['energy'],
            "sentiment": movement['sentiment'],
            "movement_score": float(movement['movement_score']),
            "velocity": float(movement['velocity'])
        },
        "emotion": {
            "emotion": emotion['emotion'],
            "sentiment": emotion['sentiment'],
            "confidence": int(emotion['confidence']),
            "details": emotion['details'],
            "all_emotions": {k: float(v) for k, v in emotion.get('all_emotions', {}).items()}
        },
        "activities": activities
    }
    
    return frame_data


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events"""
    global gemini_client, tts_client, pose_detector, posture_analyzer, video_meet_manager
    
    logger.info("=" * 60)
    logger.info("🚀 Starting AI Video Coach Backend with Meeting Support")
    logger.info("=" * 60)
    
    # Initialize pose detector
    logger.info("🎥 Initializing pose detector...")
    model_file = "openpose/models/pose/coco/pose_iter_440000.caffemodel"
    config_file = "openpose/models/pose/coco/pose_deploy_linevec.prototxt"
    
    if not os.path.exists(model_file):
        logger.error(f"Model not found at {model_file}")
        raise RuntimeError("Pose detection model not found")
    
    pose_detector = PoseDetector(model_file, config_file, use_cuda=False)
    posture_analyzer = PostureAnalyzer()
    
    # Initialize Gemini client
    gemini_client = GeminiClient()
    await gemini_client.connect()
    
    # Initialize TTS client
    tts_client = TTSClient()
    await tts_client.connect()
    
    # Initialize video meet manager
    video_meet_manager = VideoMeetManager(base_url="http://localhost:8000")
    
    logger.info("✅ All services initialized")
    logger.info("📡 Backend running with meeting support")
    logger.info("🌐 Available endpoints:")
    logger.info("   → Create Meeting: POST /api/create-meeting")
    logger.info("   → Join Meeting: GET /meet/{session_id}")
    logger.info("   → Video Analysis: ws://localhost:8000/ws/video-analysis")
    logger.info("   → Meeting Stream: ws://localhost:8000/ws/meet/{session_id}")
    
    yield
    
    # Cleanup
    logger.info("🔄 Shutting down services...")
    await gemini_client.disconnect()
    await tts_client.disconnect()
    logger.info("👋 Shutdown complete")


app = FastAPI(
    title="AI Video Coach Backend with Meeting Support",
    description="Creates video meetings with AI coaching integration",
    version="2.1.0-meetings",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "mode": "headless_backend_with_meetings",
        "service": "AI Video Coach Backend",
        "version": "2.1.0",
        "description": "Create AI coaching video meetings",
        "endpoints": {
            "create_meeting": "POST /api/create-meeting",
            "get_meeting": "GET /api/meeting/{session_id}",
            "join_meeting": "GET /meet/{session_id}",
            "list_meetings": "GET /api/meetings",
            "end_meeting": "POST /api/end-meeting/{session_id}",
            "video_analysis": "ws://localhost:8000/ws/video-analysis",
            "meeting_stream": "ws://localhost:8000/ws/meet/{session_id}",
            "health": "/health",
            "stats": "/stats"
        }
    }


@app.post("/api/create-meeting")
async def create_meeting(host_id: Optional[str] = None):
    """
    Create a new AI video coach meeting session
    
    Args:
        host_id: Optional host identifier
        
    Returns:
        Meeting details with shareable link
    
    Example response:
    {
        "success": true,
        "session_id": "abc123-def456-...",
        "meeting_link": "http://localhost:8000/meet/abc123-def456-...",
        "ws_endpoint": "ws://localhost:8000/ws/meet/abc123-def456-...",
        "share_message": "Join AI Video Coach Session: ..."
    }
    """
    try:
        # Cleanup expired sessions first
        video_meet_manager.cleanup_expired()
        
        # Create new session
        session_data = video_meet_manager.create_session(host_id)
        
        logger.info(f"📹 Meeting created: {session_data['meeting_link']}")
        
        return session_data
        
    except Exception as e:
        logger.error(f"❌ Error creating meeting: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/meeting/{session_id}")
async def get_meeting_info(session_id: str):
    """
    Get information about a meeting session
    
    Args:
        session_id: Meeting session ID
        
    Returns:
        Session details
    """
    session = video_meet_manager.get_session(session_id)
    
    if not session:
        raise HTTPException(
            status_code=404, 
            detail="Meeting not found or expired"
        )
    
    return {
        "success": True,
        "session": session.to_dict(),
        "ws_endpoint": f"ws://localhost:8000/ws/meet/{session_id}"
    }


@app.get("/meet/{session_id}")
async def join_meeting(session_id: str):
    """
    Meeting join page endpoint
    Frontend can use this to verify meeting exists
    
    Args:
        session_id: Meeting session ID
    """
    session = video_meet_manager.get_session(session_id)
    
    if not session:
        raise HTTPException(
            status_code=404,
            detail="Meeting not found or has expired"
        )
    
    return {
        "success": True,
        "message": "Meeting is ready",
        "session_id": session_id,
        "ws_endpoint": f"ws://localhost:8000/ws/meet/{session_id}",
        "redirect_to_frontend": f"http://localhost:3000/meet/{session_id}"
    }


@app.websocket("/ws/meet/{session_id}")
async def meet_websocket_endpoint(websocket: WebSocket, session_id: str):
    """
    WebSocket endpoint for meeting sessions
    Combines video analysis with real-time AI coaching
    """
    # Verify session exists
    session = video_meet_manager.get_session(session_id)
    
    if not session:
        await websocket.close(code=1008, reason="Meeting not found or expired")
        return
    
    await websocket.accept()
    
    # Generate participant ID
    participant_id = f"participant_{uuid.uuid4().hex[:8]}"
    
    # Add to session
    video_meet_manager.add_participant(session_id, participant_id)
    
    logger.info(f"👤 Participant {participant_id} joined meeting {session_id}")
    
    # Create coaching session for this participant
    coaching_session = session_manager.create_session(id(websocket))
    coach = CoachEngine(coaching_session, gemini_client, tts_client)
    
    frame_count = 0
    
    try:
        # Send welcome message
        await websocket.send_json({
            "type": "welcome",
            "message": "Connected to AI Video Coach",
            "session_id": session_id,
            "participant_id": participant_id
        })
        
        while True:
            # Receive frame from frontend
            data = await websocket.receive_text()
            message = json.loads(data)
            
            msg_type = message.get("type", "frame")
            
            if msg_type == "frame":
                # Process video frame
                frame_base64 = message.get("frame")
                if not frame_base64:
                    continue
                
                frame_count += 1
                
                # Decode and process frame
                frame = decode_base64_frame(frame_base64)
                if frame is None:
                    continue
                
                frame_data = process_frame(frame, frame_count)
                
                # Update session
                coaching_session.add_frame(frame_data)
                coaching_session.update_metrics(frame_data)
                
                # Check for coaching feedback
                if frame_count % 3 == 0:
                    should_coach, reason = await coach.should_provide_feedback(frame_data)
                    
                    if should_coach:
                        await coach.provide_feedback(frame_data, reason)
                        frame_data["coaching"] = {
                            "triggered": True,
                            "reason": reason
                        }
                
                # Send analysis back
                await websocket.send_json({
                    "type": "analysis",
                    "data": convert_to_serializable(frame_data)
                })
            
            elif msg_type == "ping":
                await websocket.send_json({"type": "pong"})
            
            elif msg_type == "end":
                break
                
    except WebSocketDisconnect:
        logger.info(f"👋 Participant {participant_id} left meeting {session_id}")
    except Exception as e:
        logger.error(f"❌ Error in meeting {session_id}: {e}")
    finally:
        video_meet_manager.remove_participant(session_id, participant_id)
        session_manager.remove_session(id(websocket))


@app.post("/api/end-meeting/{session_id}")
async def end_meeting(session_id: str):
    """End a meeting session"""
    session = video_meet_manager.get_session(session_id)
    
    if not session:
        raise HTTPException(status_code=404, detail="Meeting not found")
    
    video_meet_manager.end_session(session_id)
    
    return {
        "success": True,
        "message": f"Meeting {session_id} has been ended"
    }


@app.get("/api/meetings")
async def list_meetings():
    """List all active meetings"""
    return {
        "success": True,
        "meetings": video_meet_manager.get_all_sessions()
    }


@app.websocket("/ws/video-analysis")
async def video_analysis_endpoint(websocket: WebSocket):
    """
    Original WebSocket endpoint for direct video analysis
    (Kept for backward compatibility)
    """
    await websocket.accept()
    session_id = id(websocket)
    
    logger.info(f"📹 Frontend connected: session {session_id}")
    
    session = session_manager.create_session(session_id)
    coach = CoachEngine(session, gemini_client, tts_client)
    
    frame_count = 0
    
    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            
            frame_base64 = message.get("frame")
            if not frame_base64:
                await websocket.send_json({"error": "No frame data"})
                continue
            
            frame_count += 1
            
            frame = decode_base64_frame(frame_base64)
            if frame is None:
                await websocket.send_json({"error": "Failed to decode frame"})
                continue
            
            frame_data = process_frame(frame, frame_count)
            
            session.add_frame(frame_data)
            session.update_metrics(frame_data)
            
            if frame_count % 3 == 0:
                should_coach, reason = await coach.should_provide_feedback(frame_data)
                
                if should_coach:
                    await coach.provide_feedback(frame_data, reason)
                    frame_data["coaching"] = {
                        "triggered": True,
                        "reason": reason
                    }
            
            serializable_data = convert_to_serializable(frame_data)
            await websocket.send_json(serializable_data)
                
    except WebSocketDisconnect:
        logger.info(f"📴 Frontend disconnected: session {session_id}")
    except Exception as e:
        logger.error(f"❌ Error in session {session_id}: {e}")
    finally:
        session_manager.remove_session(session_id)


@app.get("/health")
async def health_check():
    """Detailed health check"""
    return {
        "api": "healthy",
        "mode": "headless_backend_with_meetings",
        "pose_detector": pose_detector is not None,
        "gemini": gemini_client.is_connected() if gemini_client else False,
        "tts": tts_client.is_connected() if tts_client else False,
        "active_sessions": session_manager.get_session_count(),
        "active_meetings": len(video_meet_manager.get_all_sessions())
    }


@app.get("/stats")
async def get_stats():
    """Get current system statistics"""
    return {
        "mode": "headless_backend_with_meetings",
        "active_sessions": session_manager.get_session_count(),
        "active_meetings": len(video_meet_manager.get_all_sessions()),
        "session_details": session_manager.get_all_stats(),
        "meetings": video_meet_manager.get_all_sessions(),
        "services": {
            "pose_detection": pose_detector is not None,
            "gemini_ai": gemini_client.is_connected() if gemini_client else False,
            "tts": tts_client.is_connected() if tts_client else False
        }
    }


def main():
    """Main entry point"""
    logger.info("=" * 60)
    logger.info("AI Video Coach Backend with Meeting Support")
    logger.info("=" * 60)
    logger.info("")
    logger.info("📝 Setup Instructions:")
    logger.info("1. Ensure OpenPose models are in: openpose/models/pose/coco/")
    logger.info("2. Set GEMINI_API_KEY in .env file")
    logger.info("3. Start backend: python main.py")
    logger.info("")
    logger.info("🎥 Create a meeting:")
    logger.info("   POST http://localhost:8000/api/create-meeting")
    logger.info("   Returns: meeting_link to share with participants")
    logger.info("")
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info"
    )


if __name__ == "__main__":
    main()