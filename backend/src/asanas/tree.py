"""
Tree Pose (Vrksasana) Asana Definition

Biomechanical Specification:
- Standing on one leg (standing leg straight)
- Other foot placed on inner thigh or calf (NOT on knee)
- Hips level and square forward
- Hands in prayer position or overhead
- Spine vertical
- Balance and focus
"""

from typing import Dict, Tuple
from src.services.asana_base import AsanaBase, JointAngleConstraint, AlignmentRule, JointPriority
import numpy as np


class TreePose(AsanaBase):
    """
    Tree Pose (Vrksasana) - Balance pose
    
    Critical alignment points:
    1. Standing leg straight and strong (CRITICAL - stability)
    2. Hips level (HIGH - prevents injury)
    3. Spine vertical (HIGH - balance)
    4. Lifted knee open to side (MEDIUM - hip opening)
    """
    
    def __init__(self, standing_leg='right'):
        """
        Args:
            standing_leg: 'right' or 'left' - which leg is on the ground
        """
        super().__init__()
        
        self.name = "Tree Pose"
        self.sanskrit_name = "Vrksasana"
        self.standing_leg = standing_leg
        
        # Determine which leg is standing vs lifted
        if standing_leg == 'right':
            self.standing_knee = 'right_knee'
            self.standing_hip = 'right_hip'
            self.lifted_knee = 'left_knee'
            self.lifted_hip = 'left_hip'
        else:
            self.standing_knee = 'left_knee'
            self.standing_hip = 'left_hip'
            self.lifted_knee = 'right_knee'
            self.lifted_hip = 'right_hip'
        
        self.required_joints = [
            self.standing_knee,
            self.standing_hip,
            self.lifted_knee,
            self.lifted_hip
        ]
        
        # Joint angle constraints
        self.angle_constraints = {
            # Standing leg should be straight
            self.standing_knee: JointAngleConstraint(
                joint_name=self.standing_knee,
                min_angle=165,
                max_angle=180,
                ideal_angle=175,
                tolerance=8,
                priority=JointPriority.CRITICAL
            ),
            
            # Standing hip should be neutral
            self.standing_hip: JointAngleConstraint(
                joint_name=self.standing_hip,
                min_angle=165,
                max_angle=180,
                ideal_angle=175,
                tolerance=10,
                priority=JointPriority.HIGH
            ),
            
            # Lifted knee should be bent (foot on thigh/calf)
            self.lifted_knee: JointAngleConstraint(
                joint_name=self.lifted_knee,
                min_angle=30,
                max_angle=90,
                ideal_angle=60,
                tolerance=20,
                priority=JointPriority.MEDIUM
            ),
            
            # Lifted hip - external rotation
            self.lifted_hip: JointAngleConstraint(
                joint_name=self.lifted_hip,
                min_angle=60,
                max_angle=120,
                ideal_angle=90,
                tolerance=20,
                priority=JointPriority.MEDIUM
            )
        }
        
        self.alignment_rules = [
            AlignmentRule(
                rule_id='hips_level',
                description='Hips should be level',
                check_function='check_hips_level',
                priority=JointPriority.HIGH,
                error_message='Keep your hips level and square forward'
            ),
            AlignmentRule(
                rule_id='spine_vertical',
                description='Spine should be vertical',
                check_function='check_spine_vertical',
                priority=JointPriority.HIGH,
                error_message='Lengthen your spine and stand tall'
            ),
            AlignmentRule(
                rule_id='standing_foot_grounded',
                description='Standing foot should be firmly grounded',
                check_function='check_standing_foot',
                priority=JointPriority.CRITICAL,
                error_message='Press firmly through your standing foot'
            )
        ]
        
        self.common_errors = {
            f'{self.standing_knee}_too_closed': 'Straighten your standing leg completely',
            f'{self.lifted_knee}_too_closed': 'Your lifted knee can bend more',
            f'{self.lifted_knee}_too_open': 'Bring your lifted foot higher on the thigh',
            'hips_level': 'Level your hips and avoid tilting to one side',
            'spine_vertical': 'Stand tall with your spine elongated',
            'standing_foot_grounded': 'Root down through all four corners of your standing foot'
        }
    
    def check_hips_level(self, keypoints: Dict[str, Tuple[float, float, float]]) -> Tuple[bool, float]:
        """Check if hips are level (critical for balance)"""
        if 'RHip' not in keypoints or 'LHip' not in keypoints:
            return True, 0.0
        
        r_hip = keypoints['RHip']
        l_hip = keypoints['LHip']
        
        height_diff = abs(r_hip[1] - l_hip[1])
        hip_width = abs(r_hip[0] - l_hip[0])
        
        if hip_width < 10:
            return True, 0.0
        
        # Stricter threshold for Tree Pose
        threshold = hip_width * 0.08
        
        if height_diff > threshold:
            severity = min(height_diff / (hip_width * 0.2), 1.0)
            return False, severity
        
        return True, 0.0
    
    def check_spine_vertical(self, keypoints: Dict[str, Tuple[float, float, float]]) -> Tuple[bool, float]:
        """Check if spine is vertical"""
        if 'Neck' not in keypoints or 'RHip' not in keypoints or 'LHip' not in keypoints:
            return True, 0.0
        
        neck = keypoints['Neck']
        r_hip = keypoints['RHip']
        l_hip = keypoints['LHip']
        
        mid_hip = ((r_hip[0] + l_hip[0]) / 2, (r_hip[1] + l_hip[1]) / 2)
        spine_vector = (neck[0] - mid_hip[0], neck[1] - mid_hip[1])
        
        angle = abs(np.degrees(np.arctan2(spine_vector[0], -spine_vector[1])))
        threshold = 12
        
        if angle > threshold:
            severity = min(angle / 35, 1.0)
            return False, severity
        
        return True, 0.0
    
    def check_standing_foot(self, keypoints: Dict[str, Tuple[float, float, float]]) -> Tuple[bool, float]:
        """
        Check if standing foot is grounded
        (In 2D, we check if ankle is stable relative to knee)
        """
        standing_ankle = 'RAnkle' if self.standing_leg == 'right' else 'LAnkle'
        standing_knee_kp = 'RKnee' if self.standing_leg == 'right' else 'LKnee'
        
        if standing_ankle not in keypoints or standing_knee_kp not in keypoints:
            return True, 0.0
        
        ankle = keypoints[standing_ankle]
        knee = keypoints[standing_knee_kp]
        
        # Ankle should be roughly below knee (vertical alignment)
        horizontal_offset = abs(ankle[0] - knee[0])
        vertical_distance = abs(ankle[1] - knee[1])
        
        if vertical_distance < 10:
            return True, 0.0
        
        threshold = vertical_distance * 0.15
        
        if horizontal_offset > threshold:
            severity = min(horizontal_offset / (vertical_distance * 0.3), 1.0)
            return False, severity
        
        return True, 0.0
