"""
Body Science Module
Advanced body biomechanics and analysis calculations
"""

import numpy as np


class BodyScience:
    """Calculate advanced body science metrics"""
    
    @staticmethod
    def calculate_angle(p1, p2, p3):
        """Calculate angle at p2 formed by p1-p2-p3"""
        if not all([p1, p2, p3]):
            return None
        
        a = np.array(p1[:2])
        b = np.array(p2[:2])
        c = np.array(p3[:2])
        
        ba = a - b
        bc = c - b
        
        cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
        angle = np.degrees(np.arccos(np.clip(cos_angle, -1, 1)))
        return angle
    
    @staticmethod
    def calculate_distance(p1, p2):
        """Calculate Euclidean distance between two points"""
        if not p1 or not p2:
            return None
        return np.linalg.norm(np.array(p1[:2]) - np.array(p2[:2]))
    
    @staticmethod
    def analyze_joints(points):
        """Analyze all joint angles"""
        joints = {
            'right_elbow': BodyScience.calculate_angle(points[2], points[3], points[4]),  # RShoulder-RElbow-RWrist
            'left_elbow': BodyScience.calculate_angle(points[5], points[6], points[7]),   # LShoulder-LElbow-LWrist
            'right_knee': BodyScience.calculate_angle(points[8], points[9], points[10]),  # RHip-RKnee-RAnkle
            'left_knee': BodyScience.calculate_angle(points[11], points[12], points[13]), # LHip-LKnee-LAnkle
            'right_hip': BodyScience.calculate_angle(points[2], points[8], points[9]),    # RShoulder-RHip-RKnee
            'left_hip': BodyScience.calculate_angle(points[5], points[11], points[12]),   # LShoulder-LHip-LKnee
        }
        return {k: v for k, v in joints.items() if v is not None}
    
    @staticmethod
    def analyze_symmetry(points):
        """Analyze left-right body symmetry"""
        symmetries = {}
        
        # Shoulder width
        if points[2] and points[5]:
            r_shoulder = np.array(points[2][:2])
            l_shoulder = np.array(points[5][:2])
            symmetries['shoulder_width'] = np.linalg.norm(r_shoulder - l_shoulder)
        
        # Hip width
        if points[8] and points[11]:
            r_hip = np.array(points[8][:2])
            l_hip = np.array(points[11][:2])
            symmetries['hip_width'] = np.linalg.norm(r_hip - l_hip)
        
        # Arm length symmetry
        if points[2] and points[4] and points[5] and points[7]:
            r_arm = BodyScience.calculate_distance(points[2], points[4])
            l_arm = BodyScience.calculate_distance(points[5], points[7])
            if r_arm and l_arm:
                symmetries['arm_symmetry'] = abs(r_arm - l_arm) / max(r_arm, l_arm) * 100
        
        # Leg length symmetry
        if points[8] and points[10] and points[11] and points[13]:
            r_leg = BodyScience.calculate_distance(points[8], points[10])
            l_leg = BodyScience.calculate_distance(points[11], points[13])
            if r_leg and l_leg:
                symmetries['leg_symmetry'] = abs(r_leg - l_leg) / max(r_leg, l_leg) * 100
        
        return symmetries
    
    @staticmethod
    def analyze_center_of_gravity(points):
        """Estimate center of gravity and balance"""
        valid_points = [p for p in points if p is not None]
        if not valid_points:
            return None
        
        coords = np.array([p[:2] for p in valid_points])
        cog = np.mean(coords, axis=0)
        
        # Check if CoG is within body bounds
        if points[8] and points[11]:  # Hips
            hip_y = (points[8][1] + points[11][1]) / 2
            if points[10] and points[13]:  # Ankles
                ankle_y = max(points[10][1], points[13][1])
                base_width = abs(points[10][0] - points[13][0]) if points[10] and points[13] else 100
                balance_score = 100 - (abs(cog[0] - (points[10][0] + points[13][0])/2) / max(base_width, 1) * 100)
                balance_score = np.clip(balance_score, 0, 100)
                
                return {'cog': cog, 'balance_score': balance_score}
        
        return {'cog': cog, 'balance_score': 50}
