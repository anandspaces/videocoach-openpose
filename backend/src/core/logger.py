"""
Logger Module
Handles simultaneous logging to terminal and file
"""

import os
from datetime import datetime


class MotionLogger:
    """Logger that writes to both terminal and file simultaneously"""
    
    def __init__(self, log_dir="logs", filename_prefix="motion_analysis"):
        """
        Initialize logger
        
        Args:
            log_dir: Directory to store log files
            filename_prefix: Prefix for log filenames
        """
        self.log_dir = log_dir
        self.filename_prefix = filename_prefix
        self.log_file = None
        
        # Create logs directory if it doesn't exist
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        
        # Create log file with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_filename = f"{self.filename_prefix}_{timestamp}.log"
        self.log_path = os.path.join(self.log_dir, log_filename)
        
        # Open log file in write mode
        self.log_file = open(self.log_path, 'w', encoding='utf-8')
        
        # Write header
        header = f"""
{'='*80}
Motion Analysis System - Session Log
Started: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
Log File: {self.log_path}
{'='*80}

"""
        self.log_file.write(header)
        self.log_file.flush()
        
        print(f"[LOG] Logging to: {self.log_path}")
    
    def log(self, message, to_terminal=True):
        """
        Log message to both file and terminal
        
        Args:
            message: Message to log
            to_terminal: Whether to also print to terminal (default: True)
        """
        if self.log_file and not self.log_file.closed:
            self.log_file.write(message + '\n')
            self.log_file.flush()  # Ensure immediate write
        
        if to_terminal:
            print(message)
    
    def log_frame_analysis(self, frame_num, points, points_names, joints, symmetry, 
                          cog_data, posture, movement, emotion, activities):
        """
        Log comprehensive frame analysis
        
        Args:
            frame_num: Frame number
            points: Detected keypoints
            points_names: Names of keypoints
            joints: Joint angle analysis
            symmetry: Body symmetry analysis
            cog_data: Center of gravity data
            posture: Posture analysis
            movement: Movement analysis
            emotion: Emotion analysis
            activities: Detected activities
        """
        output = []
        
        output.append(f"\n{'='*80}")
        output.append(f"FRAME {frame_num:04d} - COMPREHENSIVE BODY SCIENCE ANALYSIS")
        output.append(f"{'='*80}")
        
        # Keypoint positions
        output.append("\n[KEYPOINTS] KEYPOINT POSITIONS (18-Point Pose):")
        output.append("-" * 80)
        for i, point in enumerate(points):
            if point is not None:
                output.append(f"  {points_names[i]:12s} : X={point[0]:7.1f} Y={point[1]:7.1f} Confidence={point[2]:5.2f}")
        
        # Joint angles
        output.append(f"\n[JOINTS] JOINT ANGLES (Degrees):")
        output.append("-" * 80)
        for joint_name, angle in joints.items():
            if angle is not None:
                status = "LOCKED" if angle < 30 else "BENT" if angle < 120 else "EXTENDED"
                output.append(f"  {joint_name:15s} : {angle:6.1f}° [{status}]")
        
        # Body symmetry
        output.append(f"\n[SYMMETRY] BODY SYMMETRY ANALYSIS:")
        output.append("-" * 80)
        if 'shoulder_width' in symmetry:
            output.append(f"  Shoulder Width  : {symmetry['shoulder_width']:6.1f} pixels")
        if 'hip_width' in symmetry:
            output.append(f"  Hip Width       : {symmetry['hip_width']:6.1f} pixels")
        if 'arm_symmetry' in symmetry:
            asymmetry = symmetry['arm_symmetry']
            status = "PERFECT" if asymmetry < 5 else "BALANCED" if asymmetry < 15 else "UNBALANCED"
            output.append(f"  Arm Asymmetry   : {asymmetry:5.1f}% [{status}]")
        if 'leg_symmetry' in symmetry:
            asymmetry = symmetry['leg_symmetry']
            status = "PERFECT" if asymmetry < 5 else "BALANCED" if asymmetry < 15 else "UNBALANCED"
            output.append(f"  Leg Asymmetry   : {asymmetry:5.1f}% [{status}]")
        
        # Center of gravity
        output.append(f"\n[BALANCE] CENTER OF GRAVITY & BALANCE:")
        output.append("-" * 80)
        if cog_data:
            output.append(f"  CoG Position    : X={cog_data['cog'][0]:7.1f} Y={cog_data['cog'][1]:7.1f}")
            balance_status = "STABLE" if cog_data['balance_score'] > 70 else "MODERATE" if cog_data['balance_score'] > 40 else "UNSTABLE"
            output.append(f"  Balance Score   : {cog_data['balance_score']:5.1f}/100 [{balance_status}]")
        
        # Posture
        if posture:
            output.append(f"\n[POSTURE] POSTURE ANALYSIS:")
            output.append("-" * 80)
            output.append(f"  Status          : {posture['status']}")
            output.append(f"  Spine Angle     : {posture['angle']:6.1f}° from vertical")
            output.append(f"  Shoulder Align  : {'Balanced' if posture['shoulder_aligned'] else 'Unbalanced' if posture['shoulder_aligned'] is not None else 'Unknown'}")
        
        # Movement
        output.append(f"\n[MOVEMENT] MOVEMENT & DYNAMICS:")
        output.append("-" * 80)
        output.append(f"  Energy Level    : {movement['energy']}")
        output.append(f"  Movement Score  : {movement['movement_score']:8.2f}")
        output.append(f"  Velocity        : {movement['velocity']:6.2f} px/frame")
        output.append(f"  Sentiment       : {movement['sentiment']}")
        
        # Emotion
        output.append(f"\n[EMOTION] FACIAL EMOTION & SENTIMENT:")
        output.append("-" * 80)
        output.append(f"  Dominant        : {emotion['emotion']}")
        output.append(f"  Sentiment       : {emotion['sentiment']}")
        output.append(f"  Confidence      : {emotion['confidence']}%")
        if emotion['details']:
            output.append(f"  Distribution    : {emotion['details']}")
        
        # Activities
        output.append(f"\n[ACTIVITY] DETECTED ACTIVITIES:")
        output.append("-" * 80)
        output.append(f"  {' | '.join(activities)}")
        
        output.append(f"{'='*80}\n")
        
        # Log all lines
        for line in output:
            self.log(line)
    
    def close(self):
        """Close the log file"""
        if self.log_file and not self.log_file.closed:
            footer = f"""
{'='*80}
Session Ended: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
{'='*80}
"""
            self.log_file.write(footer)
            self.log_file.close()
            print(f"[LOG] Log saved to: {self.log_path}")
    
    def __del__(self):
        """Ensure log file is closed when object is destroyed"""
        self.close()
