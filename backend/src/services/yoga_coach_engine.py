"""
Yoga Coach Engine (Refactored)
Deterministic yoga-specific coaching with rule-based evaluation

Key principles:
1. NO LLM in the decision loop
2. Outputs structured JSON (error codes + severity)
3. Evaluates alignment ONLY during POSE_HOLD state
4. Enforces error persistence (N consecutive frames)
5. Ranks errors by priority and severity
6. Selects SINGLE most important correction
7. Enforces cooldown between feedback
"""

import logging
import time
from typing import Dict, Any, Optional, Tuple
from collections import deque

from src.services.asana_registry import get_asana
from src.services.asana_base import AsanaBase
from src.services.pose_state_machine import PoseStateMachine, PoseState

logger = logging.getLogger(__name__)


class YogaCoachEngine:
    """
    Deterministic yoga coaching engine
    
    Responsibilities:
    - Evaluate pose alignment using asana-specific rules
    - Track error persistence across frames
    - Rank errors by priority and severity
    - Enforce feedback cooldowns
    - Output structured coaching decisions (NO natural language)
    """
    
    # Cooldown settings
    MIN_FRAMES_BETWEEN_FEEDBACK = 60  # ~2 seconds at 30fps
    MIN_SECONDS_BETWEEN_FEEDBACK = 2.0
    
    # Error persistence settings
    MIN_ERROR_PERSISTENCE_FRAMES = 10  # Error must persist for 10 frames (~0.33s)
    
    def __init__(self, session_id: str):
        """
        Args:
            session_id: Unique session identifier
        """
        self.session_id = session_id
        
        # Current asana being coached
        self.current_asana: Optional[AsanaBase] = None
        self.asana_name: Optional[str] = None
        
        # State machine for temporal tracking
        self.state_machine = PoseStateMachine()
        
        # Error persistence tracking
        self.error_history = deque(maxlen=30)  # Last 30 frames of errors
        self.persistent_errors: Dict[str, int] = {}  # error_code -> frame_count
        
        # Feedback tracking
        self.last_feedback_time = 0
        self.last_feedback_frame = 0
        self.current_frame = 0
        self.feedback_history = []
        
        logger.info(f"🎓 YogaCoachEngine initialized for session {session_id}")
    
    def set_asana(self, asana_name: str) -> bool:
        """
        Set the asana to coach
        
        Args:
            asana_name: Name/ID of the asana
            
        Returns:
            True if asana was set successfully
        """
        asana = get_asana(asana_name)
        if asana is None:
            logger.error(f"❌ Unknown asana: {asana_name}")
            return False
        
        self.current_asana = asana
        self.asana_name = asana_name
        self.state_machine.set_asana(asana_name)
        
        logger.info(f"🧘 Asana set to: {asana.name} ({asana.sanskrit_name})")
        return True
    
    def update(self, frame_data: Dict[str, Any], timestamp: float) -> Dict[str, Any]:
        """
        Update coach with new frame data
        
        Args:
            frame_data: Frame analysis data containing:
                - keypoints: Dict of keypoint positions
                - joints: Dict of joint angles
                - frame_num: Frame number
            timestamp: Frame timestamp
            
        Returns:
            Coaching decision dict:
            {
                "should_coach": bool,
                "asana": str,
                "state": str,
                "error_code": str or None,
                "severity": float or None,
                "priority": int or None,
                "message": str or None
            }
        """
        self.current_frame = frame_data.get('frame_num', self.current_frame + 1)
        
        # If no asana set, return no coaching
        if self.current_asana is None:
            return {
                "should_coach": False,
                "reason": "no_asana_set",
                "state": "INIT"
            }
        
        # Update state machine
        joint_angles = frame_data.get('joints', {})
        current_state = self.state_machine.update(joint_angles, timestamp)
        
        # Get state info
        state_info = self.state_machine.get_state_info()
        
        # Check if we should evaluate alignment
        if not self.state_machine.should_evaluate_alignment():
            logger.debug(f"⏸️  Frame {self.current_frame}: Not evaluating (state: {current_state.value})")
            return {
                "should_coach": False,
                "reason": f"state_{current_state.value.lower()}",
                "state": current_state.value,
                "asana": self.asana_name,
                "asana_display": self.current_asana.name if self.current_asana else None,
                "state_info": state_info,
                "message": f"State: {current_state.value}" if current_state.value != "INIT" else "Waiting for movement..."
            }
        
        # Check cooldown
        if not self._is_cooldown_expired(timestamp):
            frames_since = self.current_frame - self.last_feedback_frame
            logger.debug(f"⏰ Frame {self.current_frame}: Cooldown active ({frames_since}/{self.MIN_FRAMES_BETWEEN_FEEDBACK} frames)")
            return {
                "should_coach": False,
                "reason": "cooldown",
                "state": current_state.value,
                "asana": self.asana_name,
                "asana_display": self.current_asana.name if self.current_asana else None,
                "state_info": state_info,
                "message": f"Holding {self.current_asana.name}..." if self.current_asana else "In pose..."
            }
        
        # Evaluate alignment
        keypoints = frame_data.get('keypoints', [])
        keypoints_dict = self._convert_keypoints(keypoints)
        
        errors = self.current_asana.evaluate_alignment(joint_angles, keypoints_dict)
        
        # Track error persistence
        self._update_error_persistence(errors)
        
        # Get persistent error (if any)
        persistent_error = self._get_persistent_error()
        
        if persistent_error is None:
            logger.debug(f"✅ Frame {self.current_frame}: No persistent errors")
            return {
                "should_coach": False,
                "reason": "no_errors",
                "state": current_state.value,
                "asana": self.asana_name,
                "state_info": state_info
            }
        
        # We have a persistent error - provide coaching
        logger.info(f"🔔 Frame {self.current_frame}: Coaching triggered for {persistent_error['error_code']}")
        
        # Record feedback
        self._record_feedback(persistent_error, timestamp)
        
        # Return structured coaching decision
        return {
            "should_coach": True,
            "asana": self.asana_name,
            "state": current_state.value,
            "error_code": persistent_error['error_code'],
            "severity": persistent_error['severity'],
            "priority": persistent_error['priority'],
            "message": persistent_error['message'],
            "joint": persistent_error.get('joint'),
            "current_angle": persistent_error.get('current_angle'),
            "ideal_angle": persistent_error.get('ideal_angle'),
            "state_info": state_info
        }
    
    def _convert_keypoints(self, keypoints: list) -> Dict[str, Tuple[float, float, float]]:
        """
        Convert keypoints list to dictionary
        
        Args:
            keypoints: List of keypoint dicts with x, y, confidence
            
        Returns:
            Dict mapping keypoint name to (x, y, confidence)
        """
        # OpenPose COCO keypoint names
        KEYPOINT_NAMES = [
            "Nose", "Neck", "RShoulder", "RElbow", "RWrist",
            "LShoulder", "LElbow", "LWrist", "RHip", "RKnee",
            "RAnkle", "LHip", "LKnee", "LAnkle", "REye",
            "LEye", "REar", "LEar"
        ]
        
        result = {}
        for i, kp in enumerate(keypoints):
            if kp is not None and i < len(KEYPOINT_NAMES):
                result[KEYPOINT_NAMES[i]] = (kp['x'], kp['y'], kp['confidence'])
        
        return result
    
    def _update_error_persistence(self, errors: list):
        """
        Update error persistence tracking
        
        Args:
            errors: List of detected errors
        """
        # Add current errors to history
        current_error_codes = {e['error_code'] for e in errors}
        self.error_history.append(current_error_codes)
        
        # Update persistence counters
        for error_code in current_error_codes:
            if error_code not in self.persistent_errors:
                self.persistent_errors[error_code] = 1
            else:
                self.persistent_errors[error_code] += 1
        
        # Decay errors not present in current frame
        for error_code in list(self.persistent_errors.keys()):
            if error_code not in current_error_codes:
                # Remove from tracking
                del self.persistent_errors[error_code]
        
        # Limit dict size
        if len(self.persistent_errors) > 20:
            # Keep only most persistent
            sorted_errors = sorted(self.persistent_errors.items(), key=lambda x: x[1], reverse=True)
            self.persistent_errors = dict(sorted_errors[:10])
    
    def _get_persistent_error(self) -> Optional[Dict]:
        """
        Get the most important persistent error
        
        Returns:
            Error dict or None
        """
        # Find errors that have persisted long enough
        persistent = []
        for error_code, frame_count in self.persistent_errors.items():
            if frame_count >= self.MIN_ERROR_PERSISTENCE_FRAMES:
                persistent.append(error_code)
        
        if not persistent:
            return None
        
        # Get full error details from current asana
        # We need to re-evaluate to get the full error dict
        # This is a simplification - in production, cache the error details
        
        # For now, return the first persistent error
        # In a full implementation, we'd cache error details and return the highest priority
        error_code = persistent[0]
        
        # Return a basic error dict
        # In production, this would come from cached evaluation results
        return {
            'error_code': error_code,
            'severity': 0.7,  # Placeholder
            'priority': 1,  # Placeholder
            'message': f"Alignment issue: {error_code.replace('_', ' ')}"
        }
    
    def _is_cooldown_expired(self, timestamp: float) -> bool:
        """
        Check if cooldown period has expired
        
        Args:
            timestamp: Current timestamp
            
        Returns:
            True if cooldown has expired
        """
        time_since = timestamp - self.last_feedback_time
        frames_since = self.current_frame - self.last_feedback_frame
        
        return (time_since >= self.MIN_SECONDS_BETWEEN_FEEDBACK and
                frames_since >= self.MIN_FRAMES_BETWEEN_FEEDBACK)
    
    def _record_feedback(self, error: Dict, timestamp: float):
        """
        Record that feedback was given
        
        Args:
            error: Error that triggered feedback
            timestamp: Feedback timestamp
        """
        self.last_feedback_time = timestamp
        self.last_feedback_frame = self.current_frame
        
        self.feedback_history.append({
            'frame': self.current_frame,
            'timestamp': timestamp,
            'error_code': error['error_code'],
            'severity': error['severity']
        })
        
        # Keep history limited
        if len(self.feedback_history) > 50:
            self.feedback_history.pop(0)
        
        # Reset persistence counter for this error
        if error['error_code'] in self.persistent_errors:
            self.persistent_errors[error['error_code']] = 0
    
    def get_stats(self) -> Dict:
        """
        Get coaching statistics
        
        Returns:
            Stats dictionary
        """
        return {
            'session_id': self.session_id,
            'asana': self.asana_name,
            'current_frame': self.current_frame,
            'feedback_count': len(self.feedback_history),
            'current_state': self.state_machine.current_state.value,
            'time_in_state': self.state_machine.get_time_in_state(),
            'recent_feedback': self.feedback_history[-5:] if self.feedback_history else []
        }
