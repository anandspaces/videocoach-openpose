"""
Video Meet Session Module
Creates video meeting sessions and returns shareable links
"""

import uuid
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
from fastapi import HTTPException
from dotenv import load_dotenv
import os

load_dotenv()
logger = logging.getLogger(__name__)

MEET_BASE_URL = os.getenv("MEET_BASE_URL", "http://localhost:8000")
WEBSOCKET_BASE_URL = os.getenv("WEBSOCKET_BASE_URL", "ws://localhost:8000")

class MeetSession:
    """Represents a video meet session"""
    
    def __init__(self, session_id: str, host_id: str):
        self.session_id = session_id
        self.host_id = host_id
        self.created_at = datetime.now()
        self.expires_at = self.created_at + timedelta(hours=2)
        self.participants = []
        self.active = True
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert session to dictionary"""
        return {
            "session_id": self.session_id,
            "host_id": self.host_id,
            "created_at": self.created_at.isoformat(),
            "expires_at": self.expires_at.isoformat(),
            "participants": self.participants,
            "active": self.active
        }
    
    def is_expired(self) -> bool:
        """Check if session has expired"""
        return datetime.now() > self.expires_at


class VideoMeetManager:
    """Manages video meet sessions"""
    
    def __init__(self, base_url: str = MEET_BASE_URL):
        self.base_url = base_url
        self.sessions: Dict[str, MeetSession] = {}
        logger.info("🎥 VideoMeetManager initialized")
    
    def create_session(self, host_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Create a new video meet session
        
        Args:
            host_id: Optional host identifier
            
        Returns:
            Dictionary with session details and meeting link
        """
        # Generate unique session ID
        session_id = str(uuid.uuid4())
        
        # Use provided host_id or generate one
        if not host_id:
            host_id = f"host_{uuid.uuid4().hex[:8]}"
        
        # Create session
        session = MeetSession(session_id, host_id)
        self.sessions[session_id] = session
        
        # Generate meeting link
        meeting_link = f"{self.base_url}/meet/{session_id}"
        
        # Generate WebSocket endpoint for this session
        ws_endpoint = f"{WEBSOCKET_BASE_URL}/ws/meet/{session_id}"
        
        logger.info(f"✅ Created video meet session: {session_id}")
        
        return {
            "success": True,
            "session_id": session_id,
            "host_id": host_id,
            "meeting_link": meeting_link,
            "ws_endpoint": ws_endpoint,
            "created_at": session.created_at.isoformat(),
            "expires_at": session.expires_at.isoformat(),
            "duration_minutes": 120,
            "share_message": f"Join AI Video Coach Session: {meeting_link}"
        }
    
    def get_session(self, session_id: str) -> Optional[MeetSession]:
        """Get session by ID"""
        session = self.sessions.get(session_id)
        
        if session and session.is_expired():
            logger.info(f"⏰ Session {session_id} has expired")
            session.active = False
            return None
        
        return session
    
    def add_participant(self, session_id: str, participant_id: str) -> bool:
        """Add participant to session"""
        session = self.get_session(session_id)
        
        if not session:
            return False
        
        if participant_id not in session.participants:
            session.participants.append(participant_id)
            logger.info(f"👤 Added participant {participant_id} to session {session_id}")
        
        return True
    
    def remove_participant(self, session_id: str, participant_id: str):
        """Remove participant from session"""
        session = self.get_session(session_id)
        
        if session and participant_id in session.participants:
            session.participants.remove(participant_id)
            logger.info(f"👋 Removed participant {participant_id} from session {session_id}")
    
    def end_session(self, session_id: str):
        """End a session"""
        if session_id in self.sessions:
            self.sessions[session_id].active = False
            logger.info(f"🛑 Ended session {session_id}")
    
    def cleanup_expired(self):
        """Remove expired sessions"""
        expired = [sid for sid, session in self.sessions.items() 
                   if session.is_expired()]
        
        for sid in expired:
            del self.sessions[sid]
            logger.info(f"🗑️ Cleaned up expired session {sid}")
        
        return len(expired)
    
    def get_all_sessions(self) -> Dict[str, Dict[str, Any]]:
        """Get all active sessions"""
        return {
            sid: session.to_dict() 
            for sid, session in self.sessions.items() 
            if session.active and not session.is_expired()
        }
