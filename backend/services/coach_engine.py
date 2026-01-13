"""
Coach Engine
Core coaching intelligence and decision making
FIXED: Enhanced logging for debugging
"""

import logging
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
    MIN_FRAMES_BETWEEN_FEEDBACK = 30  # ~1 second at 30fps (reduced for testing)
    
    # Confidence thresholds
    MIN_CONFIDENCE = 0.3
    MIN_EMOTION_CONFIDENCE = 50
    
    # Issue thresholds
    POOR_BALANCE_THRESHOLD = 40
    POOR_POSTURE_ANGLE = 40  # degrees from vertical
    HIGH_ASYMMETRY_THRESHOLD = 20  # percent difference
    
    def __init__(self, session: Any, gemini_client: Any):
        self.session = session
        self.gemini = gemini_client
        
        self.last_feedback_frame = 0
        self.consecutive_issues = {}  # Track persistent issues
        
        logger.info("🎓 CoachEngine initialized")
        
    async def should_provide_feedback(self, frame_data: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Determine if coaching feedback should be provided
        
        Args:
            frame_data: Current frame analysis
            
        Returns:
            (should_coach: bool, reason: str)
        """
        frame_num = frame_data.get("frame_num", 0)
        
        logger.debug(f"🔍 Checking frame {frame_num} for coaching opportunities")
        
        # Check cooldown
        frames_since_last = frame_num - self.last_feedback_frame
        if frames_since_last < self.MIN_FRAMES_BETWEEN_FEEDBACK:
            logger.debug(f"⏰ Cooldown active: {frames_since_last}/{self.MIN_FRAMES_BETWEEN_FEEDBACK} frames since last feedback")
            return False, ""
        
        # Check data quality
        if not self._is_high_quality_data(frame_data):
            logger.debug("⚠️ Frame data quality insufficient for coaching")
            return False, ""
        
        logger.debug("✅ Frame data quality is good")
        
        # Detect issues
        issues = self._detect_issues(frame_data)
        
        if not issues:
            logger.debug("✅ No issues detected in current frame")
            return False, ""
        
        logger.info(f"⚠️ Issues detected: {issues}")
        
        # Check if issue is persistent (appeared in multiple consecutive frames)
        persistent_issue = self._check_persistence(issues)
        
        if persistent_issue:
            logger.info(f"🔔 Persistent issue detected: {persistent_issue}")
            return True, persistent_issue
        
        logger.debug("ℹ️ Issues detected but not yet persistent")
        return False, ""
    
    async def provide_feedback(self, frame_data: Dict[str, Any], reason: str) -> str:
        """
        Generate and speak coaching feedback
        
        Args:
            frame_data: Current analysis
            reason: Reason for coaching (issue detected)
            
        Returns:
            The coaching feedback text
        """
        try:
            logger.info(f"🤖 Requesting Gemini feedback for: {reason}")
            
            # Build context for Gemini
            context = self._build_coaching_context(frame_data, reason)
            logger.debug(f"📋 Context built: {list(context.keys())}")
            
            # Get coaching feedback from Gemini
            feedback = await self.gemini.send_coaching_request(context)
            
            logger.info(f"💬 Gemini responded: {feedback}")
            
            # Update session
            self.session.record_feedback(feedback, reason)
            self.last_feedback_frame = frame_data.get("frame_num", 0)
            
            return feedback
            
        except Exception as e:
            logger.error(f"❌ Error providing feedback: {e}", exc_info=True)
            fallback = "Keep up the good work!"
            logger.warning(f"⚠️ Using fallback feedback: {fallback}")
            return fallback
    
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
        
        # Lowered threshold to 6 keypoints for partial pose analysis
        if valid_points < 6:
            logger.debug(f"❌ Only {valid_points} valid keypoints (need 6)")
            return False
        
        logger.debug(f"✅ {valid_points} valid keypoints detected")
        
        # Check emotion confidence if emotion-based coaching
        emotion = frame_data.get("emotion", {})
        if emotion.get("emotion") != "No Face":
            confidence = emotion.get("confidence", 0)
            if confidence < self.MIN_EMOTION_CONFIDENCE:
                logger.debug(f"❌ Emotion confidence too low: {confidence}% (need {self.MIN_EMOTION_CONFIDENCE}%)")
                return False
            logger.debug(f"✅ Emotion confidence: {confidence}%")
        
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
        
        # Balance issues (check if data is available)
        balance = frame_data.get("balance", {})
        balance_score = balance.get("balance_score", 100)
        if balance_score > 0 and balance_score < self.POOR_BALANCE_THRESHOLD:
            issues.append("poor_balance")
            logger.debug(f"⚠️ Poor balance detected: {balance_score:.1f}/100")
        
        # Posture issues (only check if we have posture data)
        posture = frame_data.get("posture", {})
        posture_status = posture.get("status", "Unknown")
        posture_angle = abs(posture.get("angle", 0))
        
        if posture_status != "Unknown" and posture_status != "Insufficient Data":
            if posture_angle > self.POOR_POSTURE_ANGLE:
                issues.append("poor_posture")
                logger.debug(f"⚠️ Poor posture detected: {posture_angle:.1f}° from vertical")
        
        # Symmetry issues
        symmetry = frame_data.get("symmetry", {})
        arm_asym = symmetry.get("arm_symmetry", 0)
        leg_asym = symmetry.get("leg_symmetry", 0)
        
        if arm_asym > self.HIGH_ASYMMETRY_THRESHOLD or leg_asym > self.HIGH_ASYMMETRY_THRESHOLD:
            issues.append("asymmetry")
            logger.debug(f"⚠️ Asymmetry detected: arms={arm_asym:.1f}%, legs={leg_asym:.1f}%")
        
        # Movement issues (THIS SHOULD WORK - you have movement data!)
        movement = frame_data.get("movement", {})
        energy = movement.get("energy", "")
        movement_score = movement.get("movement_score", 0)
        
        if "Very High" in energy:
            issues.append("high_energy")
            logger.debug("⚠️ Very high energy detected")
        elif "Low" in energy and self.session.get_avg_energy() > 30:
            issues.append("low_energy")
            logger.debug("⚠️ Low energy detected")
        
        # Add coaching trigger for initial movement to test the system
        if movement_score > 50:  # If there's significant movement
            issues.append("movement_detected")
            logger.debug(f"⚠️ Movement detected: score={movement_score:.1f}")
        
        # Emotion-based coaching
        emotion = frame_data.get("emotion", {})
        if emotion.get("confidence", 0) > self.MIN_EMOTION_CONFIDENCE:
            emotion_name = emotion.get("emotion", "").lower()
            
            if "sad" in emotion_name or "down" in emotion_name:
                issues.append("low_confidence")
                logger.debug(f"⚠️ Low confidence emotion: {emotion_name}")
            elif "angry" in emotion_name or "frustrated" in emotion_name:
                issues.append("frustration")
                logger.debug(f"⚠️ Frustration detected: {emotion_name}")
        
        if issues:
            logger.debug(f"📋 Total issues detected: {len(issues)}")
        
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
                logger.debug(f"📌 New issue tracked: {issue} (count=1)")
            else:
                self.consecutive_issues[issue] += 1
                logger.debug(f"📌 Issue persists: {issue} (count={self.consecutive_issues[issue]})")
        
        # Clear non-present issues
        for issue in list(self.consecutive_issues.keys()):
            if issue not in issues:
                logger.debug(f"✅ Issue resolved: {issue}")
                del self.consecutive_issues[issue]
        
        # Limit dict size to prevent memory leak
        if len(self.consecutive_issues) > 20:
            # Remove oldest entries
            sorted_issues = sorted(self.consecutive_issues.items(), key=lambda x: x[1])
            for issue, _ in sorted_issues[:10]:
                del self.consecutive_issues[issue]
        
        # Check for persistent issues (appeared in 5+ consecutive frames)
        for issue, count in self.consecutive_issues.items():
            if count >= 5:
                logger.info(f"🚨 PERSISTENT ISSUE CONFIRMED: {issue} (appeared {count} times)")
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
        context = {
            "posture": frame_data.get("posture", {}),
            "movement": frame_data.get("movement", {}),
            "emotion": frame_data.get("emotion", {}),
            "balance": frame_data.get("balance", {}),
            "symmetry": frame_data.get("symmetry", {}),
            "issue": issue,
            "session_avg_energy": self.session.get_avg_energy(),
            "session_duration": self.session.get_duration()
        }
        
        logger.debug(f"📋 Context created for issue '{issue}'")
        return context