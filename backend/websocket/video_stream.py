"""
Video Stream Handler
Processes incoming OpenPose analysis data and triggers AI coaching
"""

import json
import logging
import asyncio
from fastapi import WebSocket
from typing import Dict, Any

from services.coach_engine import CoachEngine

logger = logging.getLogger(__name__)


class VideoStreamHandler:
    """Handles incoming video analysis WebSocket stream with OpenPose data"""
    
    def __init__(self, websocket: WebSocket, session: Any, 
                 gemini_client: Any):
        self.websocket = websocket
        self.session = session
        self.gemini_client = gemini_client
        
        # Initialize coach engine
        self.coach = CoachEngine(session, gemini_client)
        
        self.frame_count = 0
        self.last_feedback_frame = 0
        
        logger.info(f"🎥 Video handler initialized for session {self.session.id}")
        
    async def handle(self):
        """Main handler loop for incoming OpenPose video data"""
        logger.info(f"🚀 Video handler started for session {self.session.id}")
        
        try:
            while True:
                # Receive frame analysis data from OpenPose
                data = await self.websocket.receive_text()
                frame_data = json.loads(data)
                
                self.frame_count += 1
                
                # Process frame
                await self.process_frame(frame_data)
                
                # Send acknowledgment back to client
                await self.websocket.send_json({
                    "status": "received",
                    "frame_num": frame_data.get("frame_num", self.frame_count)
                })
                
        except Exception as e:
            logger.error(f"Error in video handler: {e}", exc_info=True)
            raise
    
    async def process_frame(self, frame_data: Dict[str, Any]):
        """
        Process a single frame of OpenPose analysis data
        
        Args:
            frame_data: Dictionary containing comprehensive pose analysis
                - keypoints: 18-point COCO pose
                - joints: Joint angles
                - symmetry: Body symmetry metrics
                - balance: Center of gravity and balance
                - posture: Posture analysis
                - movement: Movement and energy
                - emotion: Facial emotion detection
                - activities: Detected activities
        """
        frame_num = frame_data.get("frame_num", self.frame_count)
        
        # Update session state with OpenPose data
        self.session.add_frame(frame_data)
        
        # Log key metrics periodically (every second at 30fps)
        if frame_num % 30 == 0:
            self._log_frame_summary(frame_data)
        
        # Update running session metrics
        self.session.update_metrics(frame_data)
        
        # Decide if AI coaching intervention is needed
        should_coach, reason = await self.coach.should_provide_feedback(frame_data)
        
        if should_coach:
            logger.info(f"🎯 Coaching trigger: {reason} (frame {frame_num})")
            
            # Generate and provide AI coaching feedback
            feedback = await self.coach.provide_feedback(frame_data, reason)
            
            # Send coaching feedback to client
            await self.websocket.send_json({
                "type": "coaching",
                "frame_num": frame_num,
                "reason": reason,
                "feedback": feedback,
                "timestamp": frame_data.get("timestamp")
            })
            
            self.last_feedback_frame = frame_num
    
    def _log_frame_summary(self, frame_data: Dict[str, Any]):
        """Log comprehensive frame summary from OpenPose analysis"""
        movement = frame_data.get("movement", {})
        emotion = frame_data.get("emotion", {})
        balance = frame_data.get("balance", {})
        posture = frame_data.get("posture", {})
        symmetry = frame_data.get("symmetry", {})
        
        logger.info(
            f"📊 Frame {frame_data.get('frame_num')}:\n"
            f"   Energy: {movement.get('energy', 'N/A')} "
            f"(Score: {movement.get('movement_score', 0):.1f})\n"
            f"   Emotion: {emotion.get('emotion', 'N/A')} "
            f"({emotion.get('confidence', 0)}%) - {emotion.get('sentiment', 'N/A')}\n"
            f"   Balance: {balance.get('balance_score', 0):.1f}/100\n"
            f"   Posture: {posture.get('status', 'N/A')} "
            f"(Angle: {posture.get('angle', 0):.1f}°)\n"
            f"   Arm Symmetry: {symmetry.get('arm_symmetry', 0):.1f}% diff\n"
            f"   Leg Symmetry: {symmetry.get('leg_symmetry', 0):.1f}% diff"
        )
    
    def get_session_stats(self) -> Dict[str, Any]:
        """Get current session statistics"""
        return {
            "session_id": self.session.id,
            "frames_processed": self.frame_count,
            "coaching_count": self.session.feedback_count,
            "avg_balance": self.session.get_avg_balance(),
            "avg_energy": self.session.get_avg_energy(),
            "dominant_emotion": self.session.get_dominant_emotion(),
            "duration": self.session.get_duration()
        }