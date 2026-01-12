"""
Session and State Management
Tracks user sessions and maintains state across frames
"""

import time
import logging
from collections import deque
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class Session:
    """Individual user session state"""
    
    def __init__(self, session_id: int):
        self.id = session_id
        self.start_time = time.time()
        
        # Frame buffer (keep last 30 frames for trend analysis)
        self.frame_buffer = deque(maxlen=30)
        
        # Metrics tracking
        self.total_frames = 0
        self.feedback_count = 0
        self.feedback_history = []
        
        # Running averages
        self.avg_balance = 0
        self.avg_energy = 0
        self.emotion_counts = {}
        
        logger.info(f"📊 New session created: {session_id}")
    
    def add_frame(self, frame_data: Dict[str, Any]):
        """Add frame to session buffer"""
        self.frame_buffer.append(frame_data)
        self.total_frames += 1
    
    def update_metrics(self, frame_data: Dict[str, Any]):
        """Update running session metrics"""
        # Update balance average
        balance = frame_data.get("balance", {}).get("balance_score", 0)
        self.avg_balance = (self.avg_balance * 0.9) + (balance * 0.1)
        
        # Update energy average
        movement = frame_data.get("movement", {})
        energy_score = movement.get("movement_score", 0)
        self.avg_energy = (self.avg_energy * 0.9) + (energy_score * 0.1)
        
        # Track emotions
        emotion = frame_data.get("emotion", {}).get("emotion", "Unknown")
        self.emotion_counts[emotion] = self.emotion_counts.get(emotion, 0) + 1
    
    def record_feedback(self, feedback: str, reason: str):
        """Record coaching feedback"""
        self.feedback_count += 1
        self.feedback_history.append({
            "time": time.time(),
            "feedback": feedback,
            "reason": reason,
            "frame": self.total_frames
        })
        
        # Keep only last 10 feedbacks
        if len(self.feedback_history) > 10:
            self.feedback_history.pop(0)
    
    def get_recent_frames(self, n: int = 10) -> list:
        """Get N most recent frames"""
        return list(self.frame_buffer)[-n:]
    
    def get_avg_balance(self) -> float:
        """Get average balance score"""
        return self.avg_balance
    
    def get_avg_energy(self) -> float:
        """Get average energy/movement score"""
        return self.avg_energy
    
    def get_dominant_emotion(self) -> str:
        """Get most common emotion in session"""
        if not self.emotion_counts:
            return "Unknown"
        return max(self.emotion_counts, key=self.emotion_counts.get)
    
    def get_duration(self) -> float:
        """Get session duration in seconds"""
        return time.time() - self.start_time
    
    def get_stats(self) -> Dict[str, Any]:
        """Get session statistics"""
        return {
            "session_id": self.id,
            "duration": self.get_duration(),
            "total_frames": self.total_frames,
            "feedback_count": self.feedback_count,
            "avg_balance": round(self.avg_balance, 2),
            "avg_energy": round(self.avg_energy, 2),
            "dominant_emotion": self.get_dominant_emotion(),
            "recent_feedback": self.feedback_history[-3:] if self.feedback_history else []
        }


class SessionManager:
    """Manages all active sessions"""
    
    def __init__(self):
        self.sessions: Dict[int, Session] = {}
        logger.info("📋 SessionManager initialized")
    
    def create_session(self, session_id: int) -> Session:
        """Create new session"""
        session = Session(session_id)
        self.sessions[session_id] = session
        return session
    
    def get_session(self, session_id: int) -> Optional[Session]:
        """Get session by ID"""
        return self.sessions.get(session_id)
    
    def remove_session(self, session_id: int):
        """Remove session and log final stats"""
        if session_id in self.sessions:
            session = self.sessions[session_id]
            stats = session.get_stats()
            
            logger.info(f"📊 Session {session_id} ended:")
            logger.info(f"   Duration: {stats['duration']:.1f}s")
            logger.info(f"   Frames: {stats['total_frames']}")
            logger.info(f"   Feedback given: {stats['feedback_count']}")
            logger.info(f"   Avg Balance: {stats['avg_balance']}")
            logger.info(f"   Dominant Emotion: {stats['dominant_emotion']}")
            
            del self.sessions[session_id]
    
    def get_session_count(self) -> int:
        """Get number of active sessions"""
        return len(self.sessions)
    
    def get_all_stats(self) -> Dict[int, Dict[str, Any]]:
        """Get stats for all sessions"""
        return {sid: session.get_stats() for sid, session in self.sessions.items()}