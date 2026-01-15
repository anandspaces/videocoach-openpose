"""
Asana Base Class
Defines the contract for all yoga pose definitions
"""

from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum


class JointPriority(Enum):
    """Priority levels for joint alignment checks"""
    CRITICAL = 1    # Must be correct for safety
    HIGH = 2        # Essential for pose integrity
    MEDIUM = 3      # Important for proper form
    LOW = 4         # Nice to have, refinement


@dataclass
class JointAngleConstraint:
    """Defines acceptable range for a joint angle"""
    joint_name: str
    min_angle: float  # degrees
    max_angle: float  # degrees
    ideal_angle: float  # degrees
    tolerance: float  # degrees (±)
    priority: JointPriority
    
    def is_in_range(self, angle: float) -> bool:
        """Check if angle is within acceptable range"""
        return self.min_angle <= angle <= self.max_angle
    
    def is_ideal(self, angle: float) -> bool:
        """Check if angle is within ideal tolerance"""
        return abs(angle - self.ideal_angle) <= self.tolerance
    
    def calculate_error(self, angle: float) -> float:
        """Calculate normalized error (0.0 = perfect, 1.0 = max error)"""
        if self.is_ideal(angle):
            return 0.0
        
        # Distance from ideal
        error = abs(angle - self.ideal_angle)
        
        # Normalize by range
        range_size = (self.max_angle - self.min_angle) / 2
        normalized = min(error / range_size, 1.0)
        
        return normalized


@dataclass
class AlignmentRule:
    """Defines a spatial alignment requirement"""
    rule_id: str
    description: str
    check_function: str  # Name of method to call
    priority: JointPriority
    error_message: str


class AsanaBase:
    """
    Base class for all yoga pose definitions
    
    Each asana must define:
    - Required joints
    - Joint angle constraints
    - Alignment rules
    - Common errors
    """
    
    def __init__(self):
        self.name: str = ""
        self.sanskrit_name: str = ""
        self.required_joints: List[str] = []
        self.angle_constraints: Dict[str, JointAngleConstraint] = {}
        self.alignment_rules: List[AlignmentRule] = []
        self.common_errors: Dict[str, str] = {}
    
    def validate_pose(self, joint_angles: Dict[str, float]) -> Tuple[bool, List[str]]:
        """
        Validate if current joint angles match this asana
        
        Args:
            joint_angles: Dictionary of joint_name -> angle in degrees
            
        Returns:
            (is_valid, missing_joints)
        """
        missing = []
        for joint in self.required_joints:
            if joint not in joint_angles:
                missing.append(joint)
        
        return len(missing) == 0, missing
    
    def evaluate_alignment(self, joint_angles: Dict[str, float], 
                          keypoints: Dict[str, Tuple[float, float, float]]) -> List[Dict]:
        """
        Evaluate pose alignment and detect errors
        
        Args:
            joint_angles: Dictionary of joint angles
            keypoints: Dictionary of keypoint positions (x, y, confidence)
            
        Returns:
            List of detected errors with severity scores
        """
        errors = []
        
        # Check angle constraints
        for joint_name, constraint in self.angle_constraints.items():
            if joint_name in joint_angles:
                angle = joint_angles[joint_name]
                
                if not constraint.is_in_range(angle):
                    error_severity = constraint.calculate_error(angle)
                    
                    # Determine error code
                    if angle < constraint.min_angle:
                        error_code = f"{joint_name}_too_closed"
                    else:
                        error_code = f"{joint_name}_too_open"
                    
                    errors.append({
                        'error_code': error_code,
                        'joint': joint_name,
                        'current_angle': angle,
                        'ideal_angle': constraint.ideal_angle,
                        'severity': error_severity,
                        'priority': constraint.priority.value,
                        'message': self.common_errors.get(error_code, f"{joint_name} alignment issue")
                    })
        
        # Check alignment rules
        for rule in self.alignment_rules:
            # Call the specific alignment check method
            if hasattr(self, rule.check_function):
                check_method = getattr(self, rule.check_function)
                is_aligned, severity = check_method(keypoints)
                
                if not is_aligned:
                    errors.append({
                        'error_code': rule.rule_id,
                        'joint': 'alignment',
                        'severity': severity,
                        'priority': rule.priority.value,
                        'message': rule.error_message
                    })
        
        # Sort by priority (critical first) then severity
        errors.sort(key=lambda e: (e['priority'], -e['severity']))
        
        return errors
    
    def get_top_error(self, joint_angles: Dict[str, float],
                     keypoints: Dict[str, Tuple[float, float, float]]) -> Optional[Dict]:
        """
        Get the single most important error to correct
        
        Returns:
            Error dict or None if pose is correct
        """
        errors = self.evaluate_alignment(joint_angles, keypoints)
        return errors[0] if errors else None
