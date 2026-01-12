"""
Coach Engine
Core coaching intelligence and decision making
"""

import logging
import time
from typing import Dict, Any, Tuple

logger = logging.getLogger(__name__)


class CoachEngine:
    """
    Intelligent coaching engine that:
    - Filters noise from low-confidence frames
    - Detects posture/balance/movement issues
    - Decides when to speak vs stay silent
    - Maintains coaching cooldowns
    """
    
    # Cooldown settings (in frames)
    MIN_FRAMES_BETWEEN_FEEDBACK = 90  # ~3 seconds at 30fps
    
    # Confidence thresholds
    MIN_CONFIDENCE = 0.3
    MIN_EMOTION_CONFIDENCE = 50
    
    # Issue thresholds
    POOR_BALANCE_THRESHOLD = 40
    POOR_POSTURE_ANGLE = 40  # degrees from vertical
    HIGH_ASYMMETRY_THRESHOLD = 20  # percent difference
    
    def __init__(self, session: Any, gemini_client: Any, tts_client: Any):
        self.session = session
        self.gemini = gemini_client
        self.tts = tts_client
        
        self.last_feedback_frame = 0
        self.consecutive_issues = {}  # Track persistent issues
        
    async def should_provide_feedback(self, frame_data: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Determine if coaching feedback should be provided
        
        Args:
            frame_data: Current frame analysis
            
        Returns:
            (should_coach: bool, reason: str)
        """
        frame_num = frame_data.get("frame_num", 0)
        
        # Check cooldown
        if frame_num - self.last_feedback_frame < self.MIN_FRAMES_BETWEEN_FEEDBACK:
            return False, ""
        
        # Check data quality
        if not self._is_high_quality_data(frame_data):
            return False, ""
        
        # Detect issues
        issues = self._detect_issues(frame_data)
        
        if not issues:
            return False, ""
        
        # Check if issue is persistent (appeared in multiple consecutive frames)
        persistent_issue = self._check_persistence(issues)
        
        if persistent_issue:
            return True, persistent_issue
        
        return False, ""
    
    async def provide_feedback(self, frame_data: Dict[str, Any], reason: str):
        """
        Generate and speak coaching feedback
        
        Args:
            frame_data: Current analysis
            reason: Reason for coaching (issue detected)
        """
        try:
            # Build context for Gemini
            context = self._build_coaching_context(frame_data, reason)
            
            # Get coaching feedback from Gemini
            feedback = await self.gemini.send_coaching_request(context)
            
            logger.info(f"💬 Coach says: {feedback}")
            
            # Speak feedback via TTS
            await self.tts.speak(feedback)
            
            # Update session
            self.session.record_feedback(feedback, reason)
            self.last_feedback_frame = frame_data.get("frame_num", 0)
            
        except Exception as e:
            logger.error(f"❌ Error providing feedback: {e}")
    
    def _is_high_quality_data(self, frame_data: Dict[str, Any]) -> bool:
        """
        Check if frame data is high quality enough for coaching
        
        Args:
            frame_data: Frame analysis
            
        Returns:
            True if data quality is sufficient
        """
        # Check if enough keypoints detected
        keypoints = frame_data.get("keypoints", [])
        valid_points = sum(1 for p in keypoints if p is not None)
        
        if valid_points < 10:  # Need at least 10 points for reliable analysis
            return False
        
        # Check emotion confidence if emotion-based coaching
        emotion = frame_data.get("emotion", {})
        if emotion.get("emotion") != "No Face":
            if emotion.get("confidence", 0) < self.MIN_EMOTION_CONFIDENCE:
                return False
        
        return True
    
    def _detect_issues(self, frame_data: Dict[str, Any]) -> list:
        """
        Detect posture/movement/balance issues
        
        Args:
            frame_data: Current analysis
            
        Returns:
            List of issue strings
        """
        issues = []
        
        # Balance issues
        balance = frame_data.get("balance", {})
        if balance.get("balance_score", 100) < self.POOR_BALANCE_THRESHOLD:
            issues.append("poor_balance")
        
        # Posture issues
        posture = frame_data.get("posture", {})
        posture_angle = abs(posture.get("angle", 0))
        if posture_angle > self.POOR_POSTURE_ANGLE:
            issues.append("poor_posture")
        
        # Symmetry issues
        symmetry = frame_data.get("symmetry", {})
        arm_asym = symmetry.get("arm_symmetry", 0)
        leg_asym = symmetry.get("leg_symmetry", 0)
        
        if arm_asym > self.HIGH_ASYMMETRY_THRESHOLD or leg_asym > self.HIGH_ASYMMETRY_THRESHOLD:
            issues.append("asymmetry")
        
        # Movement issues
        movement = frame_data.get("movement", {})
        energy = movement.get("energy", "")
        
        if "Very High" in energy:
            issues.append("high_energy")
        elif "Low" in energy and self.session.get_avg_energy() > 30:
            issues.append("low_energy")
        
        # Emotion-based coaching
        emotion = frame_data.get("emotion", {})
        if emotion.get("confidence", 0) > self.MIN_EMOTION_CONFIDENCE:
            emotion_name = emotion.get("emotion", "").lower()
            
            if "sad" in emotion_name or "down" in emotion_name:
                issues.append("low_confidence")
            elif "angry" in emotion_name or "frustrated" in emotion_name:
                issues.append("frustration")
        
        return issues
    
    def _check_persistence(self, issues: list) -> str:
        """
        Check if issue has been persistent across frames
        
        Args:
            issues: List of current issues
            
        Returns:
            Persistent issue name or empty string
        """
        # Update consecutive counters
        for issue in issues:
            if issue not in self.consecutive_issues:
                self.consecutive_issues[issue] = 1
            else:
                self.consecutive_issues[issue] += 1
        
        # Clear non-present issues
        for issue in list(self.consecutive_issues.keys()):
            if issue not in issues:
                del self.consecutive_issues[issue]
        
        # Check for persistent issues (appeared in 5+ consecutive frames)
        for issue, count in self.consecutive_issues.items():
            if count >= 5:
                # Reset counter to prevent immediate re-triggering
                self.consecutive_issues[issue] = 0
                return issue
        
        return ""
    
    def _build_coaching_context(self, frame_data: Dict[str, Any], issue: str) -> Dict[str, Any]:
        """
        Build context dictionary for Gemini
        
        Args:
            frame_data: Current analysis
            issue: Detected issue
            
        Returns:
            Context dictionary
        """
        return {
            "posture": frame_data.get("posture", {}),
            "movement": frame_data.get("movement", {}),
            "emotion": frame_data.get("emotion", {}),
            "balance": frame_data.get("balance", {}),
            "symmetry": frame_data.get("symmetry", {}),
            "issue": issue,
            "session_avg_energy": self.session.get_avg_energy(),
            "session_duration": self.session.get_duration()
        }