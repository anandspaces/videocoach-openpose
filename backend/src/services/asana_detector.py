"""
Asana Detector Service
Detects yoga asanas from body keypoints and joint angles using rule-based classification
"""

import logging
import time
from typing import Dict, List, Tuple, Any, Optional
from src.config.asana_definitions import (
    ASANA_DEFINITIONS,
    get_asana_names,
    get_ideal_alignment,
    get_common_mistakes,
    get_key_corrections
)

logger = logging.getLogger(__name__)


class AsanaDetector:
    """Detects yoga asanas from pose data and tracks pose state"""
    
    def __init__(self):
        self.current_asana: Optional[str] = None
        self.asana_start_time: float = 0
        self.asana_confidence: float = 0
        self.pose_history: List[Tuple[str, float]] = []  # (asana_name, timestamp)
        self.max_history = 10
        
    def detect_asana(
        self,
        keypoints: Dict[str, Dict[str, float]],
        joints: Dict[str, float],
        balance: Dict[str, Any],
        posture: Dict[str, Any]
    ) -> Tuple[Optional[str], float]:
        """
        Detect which asana the user is performing
        
        Args:
            keypoints: Dictionary of keypoint positions {point_name: {x, y, confidence}}
            joints: Dictionary of joint angles {joint_name: angle}
            balance: Balance analysis data
            posture: Posture analysis data
            
        Returns:
            (asana_name, confidence_score) or (None, 0) if no match
        """
        logger.debug(f"[ASANA_DETECTOR] Detecting asana from {len(keypoints)} keypoints and {len(joints)} joints")
        
        # Score each asana based on detection rules
        asana_scores = {}
        
        for asana_name in get_asana_names():
            asana_def = ASANA_DEFINITIONS[asana_name]
            detection_rules = asana_def.get('detection_rules', {})
            
            # Evaluate each rule
            rules_passed = 0
            total_rules = len(detection_rules)
            
            for rule_name, rule_func in detection_rules.items():
                try:
                    # Pass appropriate data to each rule
                    if 'joints' in rule_name or rule_name in ['legs_straight', 'front_knee_bent', 'back_leg_straight', 'arms_raised', 'arms_extended']:
                        result = rule_func(joints)
                    elif 'keypoints' in rule_name or 'feet' in rule_name or 'stance' in rule_name or 'hips' in rule_name:
                        result = rule_func(keypoints)
                    elif 'balance' in rule_name:
                        result = rule_func(balance)
                    elif 'posture' in rule_name or 'upright' in rule_name or 'inverted' in rule_name:
                        result = rule_func(posture)
                    else:
                        result = rule_func(keypoints)  # Default to keypoints
                    
                    if result:
                        rules_passed += 1
                        logger.debug(f"[ASANA_DETECTOR] {asana_name}.{rule_name}: PASS")
                    else:
                        logger.debug(f"[ASANA_DETECTOR] {asana_name}.{rule_name}: FAIL")
                        
                except Exception as e:
                    logger.warning(f"[ASANA_DETECTOR] Error evaluating rule {rule_name} for {asana_name}: {e}")
                    continue
            
            # Calculate confidence score (percentage of rules passed)
            if total_rules > 0:
                confidence = rules_passed / total_rules
                asana_scores[asana_name] = confidence
                logger.debug(f"[ASANA_DETECTOR] {asana_name}: {rules_passed}/{total_rules} rules passed = {confidence:.2f} confidence")
        
        # Find best match
        if not asana_scores:
            logger.info("[ASANA_DETECTOR] No asana detected (no rules evaluated)")
            return None, 0.0
        
        best_asana = max(asana_scores, key=asana_scores.get)
        best_confidence = asana_scores[best_asana]
        
        # Require minimum confidence threshold
        MIN_CONFIDENCE = 0.5  # At least 50% of rules must pass
        
        if best_confidence < MIN_CONFIDENCE:
            logger.info(f"[ASANA_DETECTOR] Best match {best_asana} ({best_confidence:.2f}) below threshold {MIN_CONFIDENCE}")
            return None, 0.0
        
        logger.info(f"[ASANA_DETECTOR] Detected: {best_asana} with {best_confidence:.2f} confidence")
        
        # Update pose tracking
        self._update_pose_tracking(best_asana, best_confidence)
        
        return best_asana, best_confidence
    
    def _update_pose_tracking(self, asana_name: str, confidence: float):
        """Update internal pose tracking state"""
        current_time = time.time()
        
        # Check if this is a new pose or continuation
        if asana_name != self.current_asana:
            logger.info(f"[ASANA_DETECTOR] Pose transition: {self.current_asana} → {asana_name}")
            self.current_asana = asana_name
            self.asana_start_time = current_time
            self.asana_confidence = confidence
        else:
            # Same pose, update confidence (running average)
            self.asana_confidence = (self.asana_confidence + confidence) / 2
        
        # Add to history
        self.pose_history.append((asana_name, current_time))
        if len(self.pose_history) > self.max_history:
            self.pose_history.pop(0)
    
    def get_pose_duration(self) -> float:
        """Get how long the current pose has been held (in seconds)"""
        if not self.current_asana or self.asana_start_time == 0:
            return 0.0
        
        duration = time.time() - self.asana_start_time
        return duration
    
    def check_pose_stability(self, min_duration: float = 2.0) -> bool:
        """
        Check if the current pose is stable (held for minimum duration)
        
        Args:
            min_duration: Minimum seconds to consider pose stable
            
        Returns:
            True if pose is stable, False otherwise
        """
        duration = self.get_pose_duration()
        is_stable = duration >= min_duration
        
        logger.debug(f"[ASANA_DETECTOR] Pose stability: {duration:.1f}s (stable={is_stable})")
        return is_stable
    
    def get_ideal_alignment_text(self, asana_name: Optional[str] = None) -> str:
        """Get ideal alignment description for current or specified asana"""
        target_asana = asana_name or self.current_asana
        if not target_asana:
            return "No asana detected"
        
        return get_ideal_alignment(target_asana)
    
    def get_common_mistakes_text(self, asana_name: Optional[str] = None) -> str:
        """Get common mistakes for current or specified asana"""
        target_asana = asana_name or self.current_asana
        if not target_asana:
            return "No asana detected"
        
        return get_common_mistakes(target_asana)
    
    def get_asana_display_name(self, asana_name: Optional[str] = None) -> str:
        """Get display name for current or specified asana"""
        target_asana = asana_name or self.current_asana
        if not target_asana:
            return "Unknown Pose"
        
        asana_def = ASANA_DEFINITIONS.get(target_asana, {})
        return asana_def.get('name', target_asana.replace('_', ' ').title())
    
    def reset(self):
        """Reset detector state"""
        logger.info("[ASANA_DETECTOR] Resetting detector state")
        self.current_asana = None
        self.asana_start_time = 0
        self.asana_confidence = 0
        self.pose_history = []
    
    def get_state(self) -> Dict[str, Any]:
        """Get current detector state for debugging"""
        return {
            'current_asana': self.current_asana,
            'display_name': self.get_asana_display_name(),
            'duration': self.get_pose_duration(),
            'confidence': self.asana_confidence,
            'is_stable': self.check_pose_stability(),
            'history_length': len(self.pose_history)
        }
