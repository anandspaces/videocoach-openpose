"""
Headless AI Video Coach System - OpenPose Integration
Server-side backend for React frontend
Integrates with OpenPose motion analysis data
FIXED: Enhanced logging for debugging
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
from concurrent.futures import ThreadPoolExecutor

# Pose detection imports
from src.core.pose_detector import PoseDetector
from src.core.posture_analyzer import PostureAnalyzer
from src.core.body_science import BodyScience

# Backend service imports
from src.services.coach_engine import CoachEngine
from src.services.state_manager import SessionManager
from src.websocket.gemini_ws import GeminiClient

# Yoga coaching system
from src.services.yoga_coach_engine import YogaCoachEngine
from src.services.asana_registry import list_asanas, get_asana

# Meeting management
from src.services.meet_session import VideoMeetManager

# Logging system
from src.core.logger import MotionLogger

from dotenv import load_dotenv

load_dotenv()

MEET_BASE_URL = os.getenv("MEET_BASE_URL")
WEBSOCKET_BASE_URL = os.getenv("WEBSOCKET_BASE_URL")

# OpenPose keypoint names (COCO 18-point model)
POSE_NAMES = [
    "Nose", "Neck", "RShoulder", "RElbow", "RWrist",
    "LShoulder", "LElbow", "LWrist", "RHip", "RKnee",
    "RAnkle", "LHip", "LKnee", "LAnkle", "REye",
    "LEye", "REar", "LEar"
]

# Configure logging with more detail
logging.basicConfig(
    level=logging.DEBUG,  # Changed to DEBUG for more visibility
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global managers
session_manager = SessionManager()
video_meet_manager = None
gemini_client = None
pose_detector = None
posture_analyzer = None
executor = ThreadPoolExecutor(max_workers=4)
session_loggers = {}  # Track loggers per session


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


def decode_base64_frame(base64_str: str) -> Optional[np.ndarray]:
    """Decode base64 image string to OpenCV frame"""
    try:
        if ',' in base64_str:
            base64_str = base64_str.split(',')[1]
        
        img_data = base64.b64decode(base64_str)
        nparr = np.frombuffer(img_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            logger.error("❌ Failed to decode frame: cv2.imdecode returned None")
        else:
            logger.debug(f"✅ Frame decoded successfully: {frame.shape}")
            
        return frame
    except Exception as e:
        logger.error(f"❌ Failed to decode frame: {e}", exc_info=True)
        return None


def _process_frame_sync(frame: np.ndarray, frame_count: int) -> Dict[str, Any]:
    """
    Synchronous frame processing using OpenPose
    """
    try:
        logger.debug(f"🔄 Processing frame {frame_count}...")
        
        # Detect pose
        points, points_prob = pose_detector.detect(frame)
        valid_keypoints = sum(1 for p in points if p is not None)
        logger.debug(f"👤 Detected {valid_keypoints} keypoints")
        
        # Analyze posture (may return None if insufficient keypoints)
        posture = posture_analyzer.analyze_posture(points)
        movement = posture_analyzer.analyze_movement(points)
        activities = posture_analyzer.detect_activity(points)
        emotion = posture_analyzer.analyze_facial_sentiment(frame, points)
        
        # Log analysis results with None checks
        posture_status = posture.get('status', 'Unknown') if posture else 'Insufficient Data'
        movement_energy = movement.get('energy', 'Unknown') if movement else 'Insufficient Data'
        logger.debug(f"📊 Posture: {posture_status}, Movement: {movement_energy}")
        
        # Body Science calculations (may also return None)
        joints = BodyScience.analyze_joints(points)
        symmetry = BodyScience.analyze_symmetry(points)
        cog_data = BodyScience.analyze_center_of_gravity(points)
        
        # Prepare analysis data with safe defaults for None values
        frame_data = {
            "frame_num": int(frame_count),
            "timestamp": float(cv2.getTickCount() / cv2.getTickFrequency()),
            "keypoints": [
                {
                    "x": float(p[0]), 
                    "y": float(p[1]), 
                    "confidence": float(p[2])
                } if p is not None else None 
                for p in points
            ],
            "joints": {k: float(v) for k, v in joints.items()} if joints else {},
            "symmetry": {k: float(v) for k, v in symmetry.items()} if symmetry else {},
            "balance": {
                "cog": [float(cog_data['cog'][0]), float(cog_data['cog'][1])],
                "balance_score": float(cog_data['balance_score'])
            } if cog_data else {"cog": [0, 0], "balance_score": 0},
            "posture": {
                "status": posture['status'],
                "angle": float(posture['angle']),
                "shoulder_aligned": bool(posture['shoulder_aligned']) if posture.get('shoulder_aligned') is not None else None
            } if posture else {"status": "Unknown", "angle": 0, "shoulder_aligned": None},
            "movement": {
                "energy": movement['energy'],
                "sentiment": movement.get('sentiment', 'Unknown'),
                "movement_score": float(movement['movement_score']),
                "velocity": float(movement['velocity'])
            } if movement else {"energy": "Unknown", "sentiment": "Unknown", "movement_score": 0, "velocity": 0},
            "emotion": {
                "emotion": emotion['emotion'],
                "sentiment": emotion['sentiment'],
                "confidence": int(emotion['confidence']),
                "details": emotion.get('details', ''),
                "all_emotions": {k: float(v) for k, v in emotion.get('all_emotions', {}).items()}
            } if emotion else {"emotion": "Unknown", "sentiment": "Unknown", "confidence": 0, "details": "", "all_emotions": {}},
            "activities": activities if activities else []
        }
        
        logger.debug(f"✅ Frame {frame_count} processed successfully")
        return frame_data
        
    except Exception as e:
        logger.error(f"❌ Error processing frame {frame_count}: {e}", exc_info=True)
        return None


async def process_frame(frame: np.ndarray, frame_count: int) -> Optional[Dict[str, Any]]:
    """
    Process a single frame asynchronously
    """
    loop = asyncio.get_event_loop()
    
    try:
        frame_data = await loop.run_in_executor(
            executor,
            _process_frame_sync,
            frame,
            frame_count
        )
        return frame_data
    except Exception as e:
        logger.error(f"❌ Error in async frame processing: {e}", exc_info=True)
        return None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events"""
    global gemini_client, pose_detector, posture_analyzer, video_meet_manager
    
    logger.info("=" * 60)
    logger.info("🚀 Starting AI Video Coach Backend with OpenPose")
    logger.info("=" * 60)
    
    # Initialize pose detector
    logger.info("🎥 Initializing OpenPose detector...")
    model_file = "src/openpose/models/pose/coco/pose_iter_440000.caffemodel"
    config_file = "src/openpose/models/pose/coco/pose_deploy_linevec.prototxt"
    
    if not os.path.exists(model_file):
        logger.error(f"Model not found at {model_file}")
        raise RuntimeError("Pose detection model not found")
    
    pose_detector = PoseDetector(model_file, config_file, use_cuda=False)
    posture_analyzer = PostureAnalyzer()
    
    # Initialize Gemini client
    logger.info("🤖 Initializing Gemini AI...")
    gemini_client = GeminiClient()
    await gemini_client.connect()
    
    # Initialize video meet manager
    video_meet_manager = VideoMeetManager(base_url=MEET_BASE_URL)
    
    logger.info("✅ All services initialized")
    logger.info(F"📡 Backend running on {MEET_BASE_URL}")
    
    yield
    
    # Cleanup
    logger.info("🔄 Shutting down services...")
    executor.shutdown(wait=True)
    await gemini_client.disconnect()
    logger.info("👋 Shutdown complete")


app = FastAPI(
    title="AI Video Coach Backend",
    description="Backend with OpenPose analysis and AI coaching",
    version="2.2.0",
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
        "service": "AI Video Coach Backend with OpenPose",
        "version": "2.2.0",
        "endpoints": {
            "create_meeting": "POST /api/create-meeting",
            "video_analysis": "{WEBSOCKET_BASE_URL}/ws/video-analysis",
            "meeting_stream": "{WEBSOCKET_BASE_URL}/ws/meet/{session_id}",
            "health": "/health",
            "stats": "/stats"
        }
    }


@app.post("/api/create-meeting")
async def create_meeting(host_id: Optional[str] = None):
    """Create a new AI video coach meeting session"""
    try:
        video_meet_manager.cleanup_expired()
        session_data = video_meet_manager.create_session(host_id)
        logger.info(f"📹 Meeting created: {session_data['meeting_link']}")
        return session_data
    except Exception as e:
        logger.error(f"Error creating meeting: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/meeting/{session_id}")
async def get_meeting_info(session_id: str):
    """Get meeting information"""
    session = video_meet_manager.get_session(session_id)
    
    if not session:
        raise HTTPException(status_code=404, detail="Meeting not found or expired")
    
    return {
        "success": True,
        "session": session.to_dict(),
        "ws_endpoint": f"{WEBSOCKET_BASE_URL}/ws/meet/{session_id}"
    }


@app.get("/meet/{session_id}")
async def join_meeting(session_id: str):
    """Meeting join endpoint"""
    session = video_meet_manager.get_session(session_id)
    
    if not session:
        raise HTTPException(status_code=404, detail="Meeting not found or expired")
    
    return {
        "success": True,
        "message": "Meeting is ready",
        "session_id": session_id,
        "ws_endpoint": f"{WEBSOCKET_BASE_URL}/ws/meet/{session_id}"
    }


@app.post("/api/end-meeting/{session_id}")
async def end_meeting(session_id: str):
    """End a meeting"""
    session = video_meet_manager.get_session(session_id)
    
    if not session:
        raise HTTPException(status_code=404, detail="Meeting not found")
    
    video_meet_manager.end_session(session_id)
    return {"success": True, "message": f"Meeting {session_id} ended"}


@app.get("/api/meetings")
async def list_meetings():
    """List all active meetings"""
    return {
        "success": True,
        "meetings": video_meet_manager.get_all_sessions()
    }


@app.get("/api/asanas")
async def list_available_asanas():
    """List all available yoga asanas"""
    asanas = list_asanas()
    return {
        "success": True,
        "asanas": asanas
    }


@app.get("/api/asana/{asana_id}")
async def get_asana_info(asana_id: str):
    """Get information about a specific asana"""
    asana = get_asana(asana_id)
    
    if not asana:
        raise HTTPException(status_code=404, detail="Asana not found")
    
    return {
        "success": True,
        "asana": {
            "id": asana_id,
            "name": asana.name,
            "sanskrit_name": asana.sanskrit_name,
            "required_joints": asana.required_joints,
            "angle_constraints": {
                joint: {
                    "min": constraint.min_angle,
                    "max": constraint.max_angle,
                    "ideal": constraint.ideal_angle,
                    "tolerance": constraint.tolerance,
                    "priority": constraint.priority.name
                }
                for joint, constraint in asana.angle_constraints.items()
            },
            "alignment_rules": [
                {
                    "id": rule.rule_id,
                    "description": rule.description,
                    "priority": rule.priority.name
                }
                for rule in asana.alignment_rules
            ]
        }
    }


@app.websocket("/ws/meet/{session_id}")
async def meet_websocket_endpoint(websocket: WebSocket, session_id: str):
    """WebSocket endpoint for meeting sessions with OpenPose coaching"""
    session = video_meet_manager.get_session(session_id)
    
    if not session:
        await websocket.close(code=1008, reason="Meeting not found")
        return
    
    await websocket.accept()
    
    # Generate unique participant ID
    participant_id = f"participant_{uuid.uuid4().hex[:8]}"
    video_meet_manager.add_participant(session_id, participant_id)
    
    logger.info(f"👤 {participant_id} joined meeting {session_id}")
    
    # Create coaching session with unique ID
    coaching_session_id = hash(f"{session_id}_{participant_id}")
    coaching_session = session_manager.create_session(coaching_session_id)
    
    # Initialize YOGA coach engine (deterministic)
    yoga_coach = YogaCoachEngine(session_id=str(coaching_session_id))
    
    # Set default asana (can be changed via message)
    yoga_coach.set_asana('tree_pose')  # Default to Tree Pose
    
    # Keep old coach for backward compatibility (optional)
    coach = CoachEngine(coaching_session, gemini_client)
    
    # Initialize session logger
    motion_logger = MotionLogger(log_dir="logs", filename_prefix=f"session_{session_id[:8]}")
    session_loggers[participant_id] = motion_logger
    
    logger.info(f"📝 Logger initialized for {participant_id}: {motion_logger.log_path}")
    motion_logger.log("="*80)
    motion_logger.log("OpenPose Motion Detection System")
    motion_logger.log("With Posture & Emotion Analysis")
    motion_logger.log("="*80)
    motion_logger.log(f"Session ID: {session_id}")
    motion_logger.log(f"Participant: {participant_id}")
    motion_logger.log("="*80)
    
    frame_count = 0
    
    try:
        await websocket.send_json({
            "type": "welcome",
            "message": "Connected to AI Video Coach with OpenPose",
            "session_id": session_id,
            "participant_id": participant_id
        })
        
        logger.info(f"✅ Welcome message sent to {participant_id}")
        
        while True:
            try:
                data = await websocket.receive_text()
                message = json.loads(data)
            except json.JSONDecodeError as e:
                logger.error(f"❌ Invalid JSON received: {e}")
                continue
            
            msg_type = message.get("type", "frame")
            logger.debug(f"📨 Received message type: {msg_type}")
            
            # Handle asana selection
            if msg_type == "set_asana":
                asana_name = message.get("asana")
                if yoga_coach.set_asana(asana_name):
                    await websocket.send_json({
                        "type": "asana_set",
                        "asana": asana_name,
                        "success": True
                    })
                    logger.info(f"🧘 Asana changed to: {asana_name}")
                else:
                    await websocket.send_json({
                        "type": "asana_set",
                        "asana": asana_name,
                        "success": False,
                        "error": "Unknown asana"
                    })
                continue
            
            if msg_type == "frame":
                frame_base64 = message.get("frame")
                if not frame_base64:
                    logger.warning("⚠️ Received frame message without frame data")
                    continue
                
                frame_count += 1
                logger.info(f"🎬 Processing frame {frame_count} from {participant_id}")
                
                # Decode frame
                frame = decode_base64_frame(frame_base64)
                if frame is None:
                    logger.error(f"❌ Failed to decode frame {frame_count}")
                    await websocket.send_json({
                        "type": "error",
                        "message": "Failed to decode frame"
                    })
                    continue
                
                # Process frame with OpenPose
                logger.debug(f"🔍 Starting OpenPose analysis for frame {frame_count}")
                frame_data = await process_frame(frame, frame_count)
                
                if frame_data is None:
                    logger.error(f"❌ Frame processing returned None for frame {frame_count}")
                    continue
                
                logger.info(f"✅ Frame {frame_count} processed successfully")
                
                # Update session
                coaching_session.add_frame(frame_data)
                coaching_session.update_metrics(frame_data)
                
                # Convert keypoints from dict to tuple format for logger
                keypoints_for_logger = []
                for kp in frame_data.get("keypoints", []):
                    if kp is not None:
                        keypoints_for_logger.append((kp['x'], kp['y'], kp['confidence']))
                    else:
                        keypoints_for_logger.append(None)
                
                # Log comprehensive frame analysis
                motion_logger.log_frame_analysis(
                    frame_num=frame_count,
                    points=keypoints_for_logger,
                    points_names=POSE_NAMES,
                    joints=frame_data.get("joints", {}),
                    symmetry=frame_data.get("symmetry", {}),
                    cog_data=frame_data.get("balance", {}),
                    posture=frame_data.get("posture", {}),
                    movement=frame_data.get("movement", {}),
                    emotion=frame_data.get("emotion", {}),
                    activities=frame_data.get("activities", [])
                )
                
                # ========================================
                # YOGA COACH SYSTEM (Deterministic)
                # ========================================
                import time
                timestamp = time.time()
                
                # Update yoga coach with frame data
                yoga_decision = yoga_coach.update(frame_data, timestamp)
                
                # Log yoga coach decision
                if yoga_decision.get('should_coach'):
                    motion_logger.log(f"\n[YOGA COACH] Frame {frame_count:04d}")
                    motion_logger.log("-" * 80)
                    motion_logger.log(f"  Asana: {yoga_decision['asana']}")
                    motion_logger.log(f"  State: {yoga_decision['state']}")
                    motion_logger.log(f"  Error: {yoga_decision['error_code']}")
                    motion_logger.log(f"  Severity: {yoga_decision['severity']:.2f}")
                    motion_logger.log(f"  Priority: {yoga_decision['priority']}")
                    motion_logger.log(f"  Message: {yoga_decision['message']}")
                    motion_logger.log("=" * 80)
                    logger.info(f"🧘 Yoga Coach: {yoga_decision['message']}")
                
                # Get Gemini response for EVERY frame (or configure interval)
                gemini_response = None
                coaching_data = None
                
                # Check every 2 frames for Gemini response (configurable)
                if frame_count % 2 == 0:
                    logger.info(f"🤖 Requesting Gemini analysis for frame {frame_count}")
                    logger.debug(f"🔧 [MAIN] Preparing context for Gemini...")
                    
                    # Prepare keypoints as a dictionary for Gemini
                    keypoints_dict = {POSE_NAMES[i]: kp for i, kp in enumerate(frame_data.get("keypoints", [])) if kp is not None}
                    logger.debug(f"🔧 [MAIN] Keypoints dict created with {len(keypoints_dict)} points")
                    logger.debug(f"🔧 [MAIN] Sample keypoint (Nose): {keypoints_dict.get('Nose', 'Not found')}")

                    # Build context for Gemini with actual movement data
                    context = {
                        "posture": frame_data.get("posture", {}),
                        "movement": frame_data.get("movement", {}),
                        "emotion": frame_data.get("emotion", {}),
                        "balance": frame_data.get("balance", {}),
                        "symmetry": frame_data.get("symmetry", {}),
                        "joints": frame_data.get("joints", {}),  # Added for specific joint feedback
                        "keypoints": keypoints_dict,  # Added for position-based feedback
                        "frame_num": frame_count
                    }
                    
                    logger.debug(f"🔧 [MAIN] Context prepared:")
                    logger.debug(f"  - Posture status: {context['posture'].get('status', 'Unknown')}")
                    logger.debug(f"  - Movement energy: {context['movement'].get('energy', 'Unknown')}")
                    logger.debug(f"  - Balance score: {context['balance'].get('balance_score', 0)}")
                    logger.debug(f"  - Joints count: {len(context['joints'])}")
                    logger.debug(f"  - Keypoints count: {len(context['keypoints'])}")
                    
                    try:
                        logger.debug("🔧 [MAIN] Calling gemini_client.send_coaching_request...")
                        # Get Gemini feedback
                        feedback = await gemini_client.send_coaching_request(context)
                        logger.debug(f"🔧 [MAIN] Gemini feedback received: {feedback}")
                        
                        gemini_response = {
                            "feedback": feedback,
                            "frame_num": frame_count,
                            "triggered": True
                        }
                        
                        # Log Gemini response
                        motion_logger.log(f"\n[GEMINI AI COACH] Frame {frame_count:04d}")
                        motion_logger.log("-" * 80)
                        motion_logger.log(f"  Response: {feedback}")
                        motion_logger.log("=" * 80)
                        
                        logger.info(f"🤖 Gemini: {feedback}")
                        
                        # Also include in coaching data for backward compatibility
                        coaching_data = {
                            "triggered": True,
                            "reason": "ai_analysis",
                            "feedback": feedback
                        }
                        
                    except Exception as e:
                        logger.error(f"❌ Gemini error: {e}")
                        logger.error(f"❌ Error type: {type(e).__name__}")
                        logger.error(f"❌ Context summary: posture={context.get('posture', {}).get('status')}, movement={context.get('movement', {}).get('energy')}")
                        gemini_response = {
                            "feedback": "Keep up the great work!",
                            "frame_num": frame_count,
                            "triggered": False,
                            "error": str(e)
                        }
                
                # Send analysis with yoga coach decision and optional Gemini
                response_data = {
                    "type": "analysis",
                    "data": convert_to_serializable(frame_data)
                }
                
                # Add YOGA COACH decision (primary coaching system)
                response_data["yoga_coach"] = yoga_decision
                
                # Add Gemini response if available (optional polishing)
                if gemini_response:
                    response_data["gemini"] = gemini_response
                
                # Add coaching data for backward compatibility
                if coaching_data:
                    response_data["coaching"] = coaching_data
                    logger.info(f"📤 Sending analysis WITH Gemini feedback")
                elif yoga_decision.get('should_coach'):
                    logger.info(f"📤 Sending analysis WITH Yoga Coach feedback")
                else:
                    logger.debug(f"📤 Sending analysis without feedback")
                
                await websocket.send_json(response_data)
            
            elif msg_type == "ping":
                logger.debug("🏓 Ping received, sending pong")
                await websocket.send_json({"type": "pong"})
            
            elif msg_type == "end":
                logger.info(f"🛑 End signal received from {participant_id}")
                break
                
    except WebSocketDisconnect:
        logger.info(f"👋 {participant_id} disconnected")
    except Exception as e:
        logger.error(f"❌ Error in meeting {session_id}: {e}", exc_info=True)
    finally:
        # Close logger
        if participant_id in session_loggers:
            session_loggers[participant_id].close()
            del session_loggers[participant_id]
            logger.info(f"📝 Logger closed for {participant_id}")
        
        video_meet_manager.remove_participant(session_id, participant_id)
        session_manager.remove_session(coaching_session_id)
        logger.info(f"🧹 Cleaned up session for {participant_id}")


@app.websocket("/ws/video-analysis")
async def video_analysis_endpoint(websocket: WebSocket):
    """Direct video analysis WebSocket endpoint with OpenPose"""
    await websocket.accept()
    
    # Create unique session ID
    session_id = hash(f"direct_{uuid.uuid4().hex[:8]}")
    logger.info(f"📹 Direct connection: session {session_id}")
    
    session = session_manager.create_session(session_id)
    coach = CoachEngine(session, gemini_client)
    
    frame_count = 0
    
    try:
        await websocket.send_json({
            "type": "connected",
            "message": "Video analysis ready with OpenPose",
            "session_id": session_id
        })
        
        while True:
            try:
                data = await websocket.receive_text()
                message = json.loads(data)
            except json.JSONDecodeError:
                await websocket.send_json({
                    "type": "error",
                    "message": "Invalid JSON"
                })
                continue
            
            msg_type = message.get("type", "frame")
            
            if msg_type == "frame":
                frame_base64 = message.get("frame")
                if not frame_base64:
                    continue
                
                frame_count += 1
                
                frame = decode_base64_frame(frame_base64)
                if frame is None:
                    continue
                
                # Process with OpenPose
                frame_data = await process_frame(frame, frame_count)
                if frame_data is None:
                    continue
                
                session.add_frame(frame_data)
                session.update_metrics(frame_data)
                
                # Check for coaching
                coaching_data = None
                if frame_count % 3 == 0:
                    should_coach, reason = await coach.should_provide_feedback(frame_data)
                    
                    if should_coach:
                        feedback = await coach.provide_feedback(frame_data, reason)
                        coaching_data = {
                            "triggered": True,
                            "reason": reason,
                            "feedback": feedback
                        }
                
                response_data = {
                    "type": "analysis",
                    "data": convert_to_serializable(frame_data)
                }
                
                if coaching_data:
                    response_data["coaching"] = coaching_data
                
                await websocket.send_json(response_data)
            
            elif msg_type == "ping":
                await websocket.send_json({"type": "pong"})
            
            elif msg_type == "end":
                break
                
    except WebSocketDisconnect:
        logger.info(f"📴 Session {session_id} disconnected")
    except Exception as e:
        logger.error(f"Error in session {session_id}: {e}", exc_info=True)
    finally:
        session_manager.remove_session(session_id)


@app.get("/health")
async def health_check():
    """Detailed health check"""
    return {
        "api": "healthy",
        "pose_detector": pose_detector is not None,
        "gemini": gemini_client.is_connected() if gemini_client else False,
        "active_sessions": session_manager.get_session_count(),
        "active_meetings": len(video_meet_manager.get_all_sessions()) if video_meet_manager else 0,
        "openpose": "loaded"
    }


@app.get("/stats")
async def get_stats():
    """System statistics"""
    return {
        "active_sessions": session_manager.get_session_count(),
        "active_meetings": len(video_meet_manager.get_all_sessions()) if video_meet_manager else 0,
        "session_details": session_manager.get_all_stats(),
        "meetings": video_meet_manager.get_all_sessions() if video_meet_manager else {},
        "services": {
            "pose_detection": "OpenPose COCO",
            "gemini_ai": gemini_client.is_connected() if gemini_client else False,
            "emotion_detection": "Haar Cascade + Analysis"
        }
    }


def main():
    """Main entry point"""
    logger.info("=" * 60)
    logger.info("AI Video Coach Backend with OpenPose")
    logger.info("=" * 60)
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
        ws_ping_interval=30,  # Send ping every 30 seconds
        ws_ping_timeout=60,   # Wait 60 seconds for pong (increased for long Gemini calls)
        timeout_keep_alive=75  # Keep connection alive for 75 seconds
    )


if __name__ == "__main__":
    main()