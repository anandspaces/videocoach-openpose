"""
Video Stream Handler
Processes incoming pose analysis data and triggers coaching
"""

import json
import logging
import asyncio
from fastapi import WebSocket
from typing import Dict, Any

from services.coach_engine import CoachEngine

logger = logging.getLogger(__name__)


class VideoStreamHandler:
    """Handles incoming video analysis WebSocket stream"""
    
    def __init__(self, websocket: WebSocket, session: Any, 
                 gemini_client: Any, tts_client: Any):
        self.websocket = websocket
        self.session = session
        self.gemini_client = gemini_client
        self.tts_client = tts_client
        
        # Initialize coach engine
        self.coach = CoachEngine(session, gemini_client, tts_client)
        
        self.frame_count = 0
        self.last_feedback_frame = 0
        
    async def handle(self):
        """Main handler loop for incoming video data"""
        logger.info(f"🎥 Video handler started for session {self.session.id}")
        
        try:
            while True:
                # Receive frame analysis data
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
            logger.error(f"Error in video handler: {e}")
            raise
    
    async def process_frame(self, frame_data: Dict[str, Any]):
        """
        Process a single frame of analysis data
        
        Args:
            frame_data: Dictionary containing pose analysis
        """
        frame_num = frame_data.get("frame_num", self.frame_count)
        
        # Update session state
        self.session.add_frame(frame_data)
        
        # Log key metrics (selective logging)
        if frame_num % 30 == 0:  # Every 30 frames (~1 second at 30fps)
            self._log_frame_summary(frame_data)
        
        # Decide if coaching intervention is needed
        should_coach, reason = await self.coach.should_provide_feedback(frame_data)
        
        if should_coach:
            logger.info(f"🎯 Coaching trigger: {reason} (frame {frame_num})")
            
            # Generate and speak coaching feedback
            await self.coach.provide_feedback(frame_data, reason)
            
            self.last_feedback_frame = frame_num
        
        # Update session metrics
        self.session.update_metrics(frame_data)
    
    def _log_frame_summary(self, frame_data: Dict[str, Any]):
        """Log frame summary for monitoring"""
        movement = frame_data.get("movement", {})
        emotion = frame_data.get("emotion", {})
        balance = frame_data.get("balance", {})
        
        logger.info(
            f"📊 Frame {frame_data.get('frame_num')}: "
            f"Energy={movement.get('energy', 'N/A')}, "
            f"Emotion={emotion.get('emotion', 'N/A')} "
            f"({emotion.get('confidence', 0)}%), "
            f"Balance={balance.get('balance_score', 0):.1f}"
        )