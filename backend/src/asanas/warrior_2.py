"""
Warrior II (Virabhadrasana II) Asana Definition

Biomechanical Specification:
- Wide stance with feet 3-4 feet apart
- Front knee bent to 90° over ankle
- Back leg straight and strong
- Arms extended parallel to ground
- Torso upright, hips square to side
- Gaze over front fingertips
"""

from typing import Dict, Tuple
from src.services.asana_base import AsanaBase, JointAngleConstraint, AlignmentRule, JointPriority
import numpy as np


class WarriorII(AsanaBase):
    """
    Warrior II Pose Definition
    
    Critical alignment points:
    1. Front knee at 90° (CRITICAL - prevents knee injury)
    2. Back leg straight (HIGH - maintains strength)
    3. Arms parallel to ground (HIGH - proper form)
    4. Hips square to side (HIGH - core engagement)
    5. Spine vertical (MEDIUM - balance)
    """
    
    def __init__(self):
        super().__init__()
        
        self.name = "Warrior II"
        self.sanskrit_name = "Virabhadrasana II"
        
        # Required joints for this pose
        self.required_joints = [
            'right_knee',
            'left_knee',
            'right_hip',
            'left_hip',
            'right_elbow',
            'left_elbow'
        ]
        
        # Joint angle constraints
        # Assuming right leg is front (will mirror for left)
        self.angle_constraints = {
            # Front knee (RIGHT) - CRITICAL
            'right_knee': JointAngleConstraint(
                joint_name='right_knee',
                min_angle=70,      # Minimum safe angle
                max_angle=110,     # Maximum before losing form
                ideal_angle=90,    # Perfect 90° bend
                tolerance=10,      # ±10° is acceptable
                priority=JointPriority.CRITICAL
            ),
            
            # Back knee (LEFT) - should be straight
            'left_knee': JointAngleConstraint(
                joint_name='left_knee',
                min_angle=160,     # Nearly straight
                max_angle=180,     # Fully straight
                ideal_angle=175,   # Almost locked
                tolerance=10,
                priority=JointPriority.HIGH
            ),
            
            # Front hip (RIGHT) - external rotation
            'right_hip': JointAngleConstraint(
                joint_name='right_hip',
                min_angle=70,
                max_angle=110,
                ideal_angle=90,
                tolerance=15,
                priority=JointPriority.HIGH
            ),
            
            # Back hip (LEFT) - slight internal rotation
            'left_hip': JointAngleConstraint(
                joint_name='left_hip',
                min_angle=160,
                max_angle=180,
                ideal_angle=170,
                tolerance=10,
                priority=JointPriority.MEDIUM
            ),
            
            # Right elbow - straight arm
            'right_elbow': JointAngleConstraint(
                joint_name='right_elbow',
                min_angle=160,
                max_angle=180,
                ideal_angle=175,
                tolerance=10,
                priority=JointPriority.MEDIUM
            ),
            
            # Left elbow - straight arm
            'left_elbow': JointAngleConstraint(
                joint_name='left_elbow',
                min_angle=160,
                max_angle=180,
                ideal_angle=175,
                tolerance=10,
                priority=JointPriority.MEDIUM
            )
        }
        
        # Alignment rules (spatial relationships)
        self.alignment_rules = [
            AlignmentRule(
                rule_id='front_knee_over_ankle',
                description='Front knee should be directly over ankle',
                check_function='check_knee_over_ankle',
                priority=JointPriority.CRITICAL,
                error_message='Align your front knee directly over your ankle'
            ),
            AlignmentRule(
                rule_id='arms_parallel_to_ground',
                description='Arms should be parallel to ground',
                check_function='check_arms_parallel',
                priority=JointPriority.HIGH,
                error_message='Extend your arms parallel to the ground'
            ),
            AlignmentRule(
                rule_id='hips_square_to_side',
                description='Hips should be square to the side',
                check_function='check_hips_square',
                priority=JointPriority.HIGH,
                error_message='Square your hips to the side of your mat'
            ),
            AlignmentRule(
                rule_id='spine_vertical',
                description='Spine should be vertical',
                check_function='check_spine_vertical',
                priority=JointPriority.MEDIUM,
                error_message='Keep your torso upright and spine vertical'
            ),
            AlignmentRule(
                rule_id='shoulders_over_hips',
                description='Shoulders should be stacked over hips',
                check_function='check_shoulders_over_hips',
                priority=JointPriority.MEDIUM,
                error_message='Stack your shoulders directly over your hips'
            )
        ]
        
        # Common error patterns and corrections
        self.common_errors = {
            'right_knee_too_closed': 'Bend your front knee deeper toward 90 degrees',
            'right_knee_too_open': 'Your front knee is over-extended, bend it to 90 degrees',
            'left_knee_too_closed': 'Straighten your back leg completely',
            'left_knee_too_open': 'Back leg is good, maintain that strength',
            'right_elbow_too_closed': 'Straighten your front arm fully',
            'left_elbow_too_closed': 'Straighten your back arm fully',
            'front_knee_over_ankle': 'Align your front knee directly over your ankle',
            'arms_parallel_to_ground': 'Lift or lower your arms to be parallel with the ground',
            'hips_square_to_side': 'Rotate your hips to face the side of your mat',
            'spine_vertical': 'Bring your torso upright, avoid leaning forward or back',
            'shoulders_over_hips': 'Stack your shoulders over your hips'
        }
    
    def check_knee_over_ankle(self, keypoints: Dict[str, Tuple[float, float, float]]) -> Tuple[bool, float]:
        """
        Check if front knee is aligned over ankle (critical for safety)
        
        Returns:
            (is_aligned, severity)
        """
        # Assuming right leg is front
        if 'RKnee' not in keypoints or 'RAnkle' not in keypoints:
            return True, 0.0  # Can't check, assume OK
        
        knee = keypoints['RKnee']
        ankle = keypoints['RAnkle']
        
        # Horizontal distance between knee and ankle
        horizontal_distance = abs(knee[0] - ankle[0])
        
        # Vertical distance (for normalization)
        vertical_distance = abs(knee[1] - ankle[1])
        
        if vertical_distance < 10:  # Too close, probably not in pose
            return True, 0.0
        
        # Knee should be within 20% of vertical distance horizontally
        threshold = vertical_distance * 0.2
        
        if horizontal_distance > threshold:
            # Calculate severity (0.0 = perfect, 1.0 = very bad)
            severity = min(horizontal_distance / (vertical_distance * 0.5), 1.0)
            return False, severity
        
        return True, 0.0
    
    def check_arms_parallel(self, keypoints: Dict[str, Tuple[float, float, float]]) -> Tuple[bool, float]:
        """
        Check if arms are parallel to ground
        
        Returns:
            (is_aligned, severity)
        """
        if 'RWrist' not in keypoints or 'LWrist' not in keypoints:
            return True, 0.0
        
        if 'RShoulder' not in keypoints or 'LShoulder' not in keypoints:
            return True, 0.0
        
        r_wrist = keypoints['RWrist']
        l_wrist = keypoints['LWrist']
        r_shoulder = keypoints['RShoulder']
        l_shoulder = keypoints['LShoulder']
        
        # Calculate arm slopes
        r_arm_slope = (r_wrist[1] - r_shoulder[1]) / (r_wrist[0] - r_shoulder[0] + 1e-6)
        l_arm_slope = (l_wrist[1] - l_shoulder[1]) / (l_wrist[0] - l_shoulder[0] + 1e-6)
        
        # Convert to angles from horizontal
        r_angle = abs(np.degrees(np.arctan(r_arm_slope)))
        l_angle = abs(np.degrees(np.arctan(l_arm_slope)))
        
        # Arms should be within 15° of horizontal
        threshold = 15
        
        max_deviation = max(r_angle, l_angle)
        
        if max_deviation > threshold:
            severity = min(max_deviation / 45, 1.0)  # Normalize to 45° max
            return False, severity
        
        return True, 0.0
    
    def check_hips_square(self, keypoints: Dict[str, Tuple[float, float, float]]) -> Tuple[bool, float]:
        """
        Check if hips are square to the side
        
        Returns:
            (is_aligned, severity)
        """
        if 'RHip' not in keypoints or 'LHip' not in keypoints:
            return True, 0.0
        
        r_hip = keypoints['RHip']
        l_hip = keypoints['LHip']
        
        # In Warrior II, hips should be roughly at same depth (y-coordinate similar)
        # This is a simplified check - in 2D we can't see true rotation
        
        hip_height_diff = abs(r_hip[1] - l_hip[1])
        hip_width = abs(r_hip[0] - l_hip[0])
        
        if hip_width < 10:  # Too narrow, probably not in pose
            return True, 0.0
        
        # Height difference should be less than 10% of width
        threshold = hip_width * 0.1
        
        if hip_height_diff > threshold:
            severity = min(hip_height_diff / (hip_width * 0.3), 1.0)
            return False, severity
        
        return True, 0.0
    
    def check_spine_vertical(self, keypoints: Dict[str, Tuple[float, float, float]]) -> Tuple[bool, float]:
        """
        Check if spine is vertical (torso upright)
        
        Returns:
            (is_aligned, severity)
        """
        if 'Neck' not in keypoints:
            return True, 0.0
        
        if 'RHip' not in keypoints or 'LHip' not in keypoints:
            return True, 0.0
        
        neck = keypoints['Neck']
        r_hip = keypoints['RHip']
        l_hip = keypoints['LHip']
        
        # Mid-hip point
        mid_hip = ((r_hip[0] + l_hip[0]) / 2, (r_hip[1] + l_hip[1]) / 2)
        
        # Spine vector
        spine_vector = (neck[0] - mid_hip[0], neck[1] - mid_hip[1])
        
        # Angle from vertical
        angle = abs(np.degrees(np.arctan2(spine_vector[0], -spine_vector[1])))
        
        # Should be within 15° of vertical
        threshold = 15
        
        if angle > threshold:
            severity = min(angle / 45, 1.0)
            return False, severity
        
        return True, 0.0
    
    def check_shoulders_over_hips(self, keypoints: Dict[str, Tuple[float, float, float]]) -> Tuple[bool, float]:
        """
        Check if shoulders are stacked over hips
        
        Returns:
            (is_aligned, severity)
        """
        if 'RShoulder' not in keypoints or 'LShoulder' not in keypoints:
            return True, 0.0
        
        if 'RHip' not in keypoints or 'LHip' not in keypoints:
            return True, 0.0
        
        r_shoulder = keypoints['RShoulder']
        l_shoulder = keypoints['LShoulder']
        r_hip = keypoints['RHip']
        l_hip = keypoints['LHip']
        
        # Mid-points
        mid_shoulder = ((r_shoulder[0] + l_shoulder[0]) / 2, (r_shoulder[1] + l_shoulder[1]) / 2)
        mid_hip = ((r_hip[0] + l_hip[0]) / 2, (r_hip[1] + l_hip[1]) / 2)
        
        # Horizontal offset
        horizontal_offset = abs(mid_shoulder[0] - mid_hip[0])
        
        # Vertical distance (for normalization)
        vertical_distance = abs(mid_shoulder[1] - mid_hip[1])
        
        if vertical_distance < 10:
            return True, 0.0
        
        # Offset should be less than 15% of vertical distance
        threshold = vertical_distance * 0.15
        
        if horizontal_offset > threshold:
            severity = min(horizontal_offset / (vertical_distance * 0.4), 1.0)
            return False, severity
        
        return True, 0.0
