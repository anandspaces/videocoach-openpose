"""
Real-Time AI Video Coach Backend
FastAPI + WebSocket + Gemini + TTS Integration
"""

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging
import asyncio

from websocket.video_stream import VideoStreamHandler
from websocket.gemini_ws import GeminiClient
from websocket.tts_ws import TTSClient
from services.coach_engine import CoachEngine
from services.state_manager import SessionManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global managers
session_manager = SessionManager()
gemini_client = None
tts_client = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events"""
    global gemini_client, tts_client
    
    logger.info("🚀 Starting AI Video Coach Backend...")
    
    # Initialize Gemini client
    gemini_client = GeminiClient()
    await gemini_client.connect()
    
    # Initialize TTS client
    tts_client = TTSClient()
    await tts_client.connect()
    
    logger.info("✅ All services initialized")
    
    yield
    
    # Cleanup
    logger.info("🔄 Shutting down services...")
    await gemini_client.disconnect()
    await tts_client.disconnect()
    logger.info("👋 Shutdown complete")


app = FastAPI(
    title="AI Video Coach API",
    description="Real-time pose analysis with AI coaching",
    version="1.0.0",
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
        "service": "AI Video Coach",
        "version": "1.0.0",
        "endpoints": {
            "video_analysis": "/ws/video-analysis",
            "coach_audio": "/ws/coach-audio",
            "health": "/health"
        }
    }


@app.get("/health")
async def health_check():
    """Detailed health check"""
    return {
        "api": "healthy",
        "gemini": gemini_client.is_connected() if gemini_client else False,
        "tts": tts_client.is_connected() if tts_client else False,
        "active_sessions": session_manager.get_session_count()
    }


@app.websocket("/ws/video-analysis")
async def video_analysis_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for receiving real-time pose analysis data
    
    Expected JSON format per frame:
    {
        "frame_num": 123,
        "timestamp": 1234567890.123,
        "keypoints": [...],
        "joints": {...},
        "symmetry": {...},
        "balance": {...},
        "movement": {...},
        "emotion": {...},
        "activities": [...]
    }
    """
    await websocket.accept()
    session_id = id(websocket)
    
    logger.info(f"📹 New video analysis session: {session_id}")
    
    # Create session
    session = session_manager.create_session(session_id)
    
    # Create handler
    handler = VideoStreamHandler(
        websocket=websocket,
        session=session,
        gemini_client=gemini_client,
        tts_client=tts_client
    )
    
    try:
        await handler.handle()
    except WebSocketDisconnect:
        logger.info(f"📴 Video session disconnected: {session_id}")
    except Exception as e:
        logger.error(f"❌ Error in video session {session_id}: {e}")
    finally:
        session_manager.remove_session(session_id)
        await websocket.close()


@app.websocket("/ws/coach-audio")
async def coach_audio_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for streaming TTS audio to frontend
    Streams audio chunks in real-time
    """
    await websocket.accept()
    session_id = id(websocket)
    
    logger.info(f"🔊 New audio stream session: {session_id}")
    
    try:
        while True:
            # Wait for audio chunks from TTS client
            if tts_client and tts_client.has_audio():
                audio_chunk = await tts_client.get_audio_chunk()
                await websocket.send_bytes(audio_chunk)
            else:
                await asyncio.sleep(0.01)  # Small delay to prevent busy waiting
                
    except WebSocketDisconnect:
        logger.info(f"📴 Audio session disconnected: {session_id}")
    except Exception as e:
        logger.error(f"❌ Error in audio session {session_id}: {e}")
    finally:
        await websocket.close()


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )