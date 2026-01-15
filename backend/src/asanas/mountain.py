"""
Mountain Pose (Tadasana) Asana Definition

Biomechanical Specification:
- Standing upright with feet together or hip-width apart
- Weight evenly distributed on both feet
- Legs straight and strong
- Arms at sides or overhead
- Spine elongated and vertical
- Shoulders relaxed
"""

from typing import Dict, Tuple
from src.services.asana_base import AsanaBase, JointAngleConstraint, AlignmentRule, JointPriority
import numpy as np


class MountainPose(AsanaBase):
    """
    Mountain Pose (Tadasana) - Foundation standing pose
    
    Critical alignment points:
    1. Spine vertical (CRITICAL - foundation of all standing poses)
    2. Weight balanced between feet (HIGH - stability)
    3. Knees straight but not locked (MEDIUM - strength without strain)
    4. Shoulders relaxed (MEDIUM - upper body alignment)
    """
    
    def __init__(self):
        super().__init__()
        
        self.name = "Mountain Pose"
        self.sanskrit_name = "Tadasana"
        
        self.required_joints = [
            'right_knee',
            'left_knee',
            'right_hip',
            'left_hip'
        ]
        
        # Joint angle constraints
        self.angle_constraints = {
            # Both knees should be straight
            'right_knee': JointAngleConstraint(
                joint_name='right_knee',
                min_angle=165,
                max_angle=180,
                ideal_angle=175,
                tolerance=8,
                priority=JointPriority.MEDIUM
            ),
            
            'left_knee': JointAngleConstraint(
                joint_name='left_knee',
                min_angle=165,
                max_angle=180,
                ideal_angle=175,
                tolerance=8,
                priority=JointPriority.MEDIUM
            ),
            
            # Hips should be neutral
            'right_hip': JointAngleConstraint(
                joint_name='right_hip',
                min_angle=165,
                max_angle=180,
                ideal_angle=175,
                tolerance=10,
                priority=JointPriority.MEDIUM
            ),
            
            'left_hip': JointAngleConstraint(
                joint_name='left_hip',
                min_angle=165,
                max_angle=180,
                ideal_angle=175,
                tolerance=10,
                priority=JointPriority.MEDIUM
            )
        }
        
        self.alignment_rules = [
            AlignmentRule(
                rule_id='spine_vertical',
                description='Spine should be vertical',
                check_function='check_spine_vertical',
                priority=JointPriority.CRITICAL,
                error_message='Lengthen your spine and stand tall'
            ),
            AlignmentRule(
                rule_id='weight_balanced',
                description='Weight evenly distributed',
                check_function='check_weight_balanced',
                priority=JointPriority.HIGH,
                error_message='Distribute your weight evenly between both feet'
            ),
            AlignmentRule(
                rule_id='shoulders_level',
                description='Shoulders should be level',
                check_function='check_shoulders_level',
                priority=JointPriority.MEDIUM,
                error_message='Level your shoulders and relax them down'
            )
        ]
        
        self.common_errors = {
            'right_knee_too_closed': 'Straighten your right leg without locking the knee',
            'left_knee_too_closed': 'Straighten your left leg without locking the knee',
            'spine_vertical': 'Stand tall with your spine elongated',
            'weight_balanced': 'Balance your weight evenly on both feet',
            'shoulders_level': 'Level your shoulders and draw them away from your ears'
        }
    
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
        threshold = 10  # Stricter for Mountain Pose
        
        if angle > threshold:
            severity = min(angle / 30, 1.0)
            return False, severity
        
        return True, 0.0
    
    def check_weight_balanced(self, keypoints: Dict[str, Tuple[float, float, float]]) -> Tuple[bool, float]:
        """Check if weight is balanced between feet"""
        if 'RAnkle' not in keypoints or 'LAnkle' not in keypoints:
            return True, 0.0
        
        if 'RHip' not in keypoints or 'LHip' not in keypoints:
            return True, 0.0
        
        r_ankle = keypoints['RAnkle']
        l_ankle = keypoints['LAnkle']
        r_hip = keypoints['RHip']
        l_hip = keypoints['LHip']
        
        # Check if hips are level (indicator of weight distribution)
        hip_height_diff = abs(r_hip[1] - l_hip[1])
        hip_width = abs(r_hip[0] - l_hip[0])
        
        if hip_width < 10:
            return True, 0.0
        
        threshold = hip_width * 0.05  # Very strict for Mountain Pose
        
        if hip_height_diff > threshold:
            severity = min(hip_height_diff / (hip_width * 0.15), 1.0)
            return False, severity
        
        return True, 0.0
    
    def check_shoulders_level(self, keypoints: Dict[str, Tuple[float, float, float]]) -> Tuple[bool, float]:
        """Check if shoulders are level"""
        if 'RShoulder' not in keypoints or 'LShoulder' not in keypoints:
            return True, 0.0
        
        r_shoulder = keypoints['RShoulder']
        l_shoulder = keypoints['LShoulder']
        
        height_diff = abs(r_shoulder[1] - l_shoulder[1])
        shoulder_width = abs(r_shoulder[0] - l_shoulder[0])
        
        if shoulder_width < 10:
            return True, 0.0
        
        threshold = shoulder_width * 0.08
        
        if height_diff > threshold:
            severity = min(height_diff / (shoulder_width * 0.2), 1.0)
            return False, severity
        
        return True, 0.0
