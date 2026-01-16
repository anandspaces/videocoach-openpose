"""
Adaptive Feedback Manager - Prevents overwhelming the user with feedback
Implements cooldown periods and priority-based feedback logic
"""

import time
from typing import Dict, Any, Optional
from enum import Enum


class FeedbackPriority(Enum):
    """Feedback priority levels"""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


class FeedbackType(Enum):
    """Types of feedback"""
    VISUAL = "visual"
    VOICE = "voice"
    BOTH = "both"


class FeedbackManager:
    """
    Manages feedback timing to prevent overwhelming the user
    Implements cooldown periods and priority-based logic
    """
    
    def __init__(
        self,
        voice_cooldown: float = 5.0,
        visual_cooldown: float = 1.0,
        critical_bypass: bool = True
    ):
        """
        Initialize feedback manager
        
        Args:
            voice_cooldown: Seconds between voice feedback (default: 5.0)
            visual_cooldown: Seconds between visual feedback updates (default: 1.0)
            critical_bypass: Allow critical errors to bypass cooldown (default: True)
        """
        self.voice_cooldown = voice_cooldown
        self.visual_cooldown = visual_cooldown
        self.critical_bypass = critical_bypass
        
        self.last_voice_feedback = 0.0
        self.last_visual_feedback = 0.0
        self.last_feedback_message = ""
        self.feedback_count = 0
    
    def should_give_feedback(
        self,
        error: Dict[str, Any],
        feedback_type: FeedbackType = FeedbackType.VOICE,
        is_user_speaking: bool = False
    ) -> bool:
        """
        Determine if feedback should be given based on timing and priority
        
        Args:
            error: Error dictionary with 'severity', 'priority', 'message' keys
            feedback_type: Type of feedback to check
            is_user_speaking: Whether user is currently speaking
            
        Returns:
            True if feedback should be given, False otherwise
        """
        now = time.time()
        
        # Never give voice feedback while user is speaking
        if is_user_speaking and feedback_type in [FeedbackType.VOICE, FeedbackType.BOTH]:
            return False
        
        # Extract priority
        priority = self._get_priority(error)
        
        # Critical errors bypass cooldown if enabled
        if self.critical_bypass and priority == FeedbackPriority.CRITICAL:
            return True
        
        # Check cooldown based on feedback type
        if feedback_type == FeedbackType.VOICE:
            time_since_last = now - self.last_voice_feedback
            return time_since_last >= self.voice_cooldown
        
        elif feedback_type == FeedbackType.VISUAL:
            time_since_last = now - self.last_visual_feedback
            return time_since_last >= self.visual_cooldown
        
        elif feedback_type == FeedbackType.BOTH:
            voice_ready = (now - self.last_voice_feedback) >= self.voice_cooldown
            visual_ready = (now - self.last_visual_feedback) >= self.visual_cooldown
            return voice_ready and visual_ready
        
        return False
    
    def record_feedback(
        self,
        feedback_type: FeedbackType,
        message: str
    ) -> None:
        """
        Record that feedback was given
        
        Args:
            feedback_type: Type of feedback given
            message: Feedback message
        """
        now = time.time()
        
        if feedback_type in [FeedbackType.VOICE, FeedbackType.BOTH]:
            self.last_voice_feedback = now
        
        if feedback_type in [FeedbackType.VISUAL, FeedbackType.BOTH]:
            self.last_visual_feedback = now
        
        self.last_feedback_message = message
        self.feedback_count += 1
    
    def _get_priority(self, error: Dict[str, Any]) -> FeedbackPriority:
        """
        Extract priority from error dictionary
        
        Args:
            error: Error dictionary
            
        Returns:
            FeedbackPriority enum value
        """
        # Try to get priority from error
        if 'priority' in error:
            priority_str = str(error['priority']).upper()
            try:
                return FeedbackPriority[priority_str]
            except KeyError:
                pass
        
        # Fallback to severity
        if 'severity' in error:
            severity = error['severity']
            if isinstance(severity, (int, float)):
                if severity >= 0.9:
                    return FeedbackPriority.CRITICAL
                elif severity >= 0.7:
                    return FeedbackPriority.HIGH
                elif severity >= 0.4:
                    return FeedbackPriority.MEDIUM
                else:
                    return FeedbackPriority.LOW
            elif isinstance(severity, str):
                severity_map = {
                    'critical': FeedbackPriority.CRITICAL,
                    'high': FeedbackPriority.HIGH,
                    'medium': FeedbackPriority.MEDIUM,
                    'low': FeedbackPriority.LOW
                }
                return severity_map.get(severity.lower(), FeedbackPriority.MEDIUM)
        
        # Default to medium priority
        return FeedbackPriority.MEDIUM
    
    def get_time_until_next_feedback(
        self,
        feedback_type: FeedbackType = FeedbackType.VOICE
    ) -> float:
        """
        Get time remaining until next feedback is allowed
        
        Args:
            feedback_type: Type of feedback to check
            
        Returns:
            Seconds until next feedback (0 if ready now)
        """
        now = time.time()
        
        if feedback_type == FeedbackType.VOICE:
            time_since = now - self.last_voice_feedback
            remaining = max(0, self.voice_cooldown - time_since)
            return remaining
        
        elif feedback_type == FeedbackType.VISUAL:
            time_since = now - self.last_visual_feedback
            remaining = max(0, self.visual_cooldown - time_since)
            return remaining
        
        elif feedback_type == FeedbackType.BOTH:
            voice_remaining = max(0, self.voice_cooldown - (now - self.last_voice_feedback))
            visual_remaining = max(0, self.visual_cooldown - (now - self.last_visual_feedback))
            return max(voice_remaining, visual_remaining)
        
        return 0.0
    
    def reset(self) -> None:
        """Reset all feedback timers"""
        self.last_voice_feedback = 0.0
        self.last_visual_feedback = 0.0
        self.last_feedback_message = ""
        self.feedback_count = 0
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get feedback statistics
        
        Returns:
            Dictionary with feedback stats
        """
        now = time.time()
        return {
            'total_feedback_count': self.feedback_count,
            'last_message': self.last_feedback_message,
            'time_since_voice': now - self.last_voice_feedback,
            'time_since_visual': now - self.last_visual_feedback,
            'voice_ready': (now - self.last_voice_feedback) >= self.voice_cooldown,
            'visual_ready': (now - self.last_visual_feedback) >= self.visual_cooldown
        }
