"""
Circular Pose Buffer - Stores last 3 seconds of pose data
Enables stability analysis and quick snapshot retrieval for voice queries
"""

import time
from typing import Dict, Any, List, Optional
from collections import deque
import numpy as np


class CircularPoseBuffer:
    """
    Circular buffer to store pose data with automatic cleanup.
    Maintains last N frames (default: 90 frames = 3 seconds at 30 FPS)
    """
    
    def __init__(self, max_size: int = 90):
        """
        Initialize circular buffer
        
        Args:
            max_size: Maximum number of frames to store (default: 90 = 3 seconds at 30 FPS)
        """
        self.buffer = deque(maxlen=max_size)
        self.max_size = max_size
    
    def push(self, pose_data: Dict[str, Any]) -> None:
        """
        Add pose data to buffer
        
        Args:
            pose_data: Dictionary containing pose analysis data
                Expected keys: timestamp, keypoints, features, errors, joints, balance, etc.
        """
        # Ensure timestamp is present
        if 'timestamp' not in pose_data:
            pose_data['timestamp'] = time.time()
        
        self.buffer.append(pose_data)
    
    def get_last_n_frames(self, n: int) -> List[Dict[str, Any]]:
        """
        Get last N frames from buffer
        
        Args:
            n: Number of frames to retrieve
            
        Returns:
            List of pose data dictionaries (most recent last)
        """
        if n >= len(self.buffer):
            return list(self.buffer)
        return list(self.buffer)[-n:]
    
    def get_latest(self) -> Optional[Dict[str, Any]]:
        """
        Get most recent frame
        
        Returns:
            Latest pose data or None if buffer is empty
        """
        if len(self.buffer) == 0:
            return None
        return self.buffer[-1]
    
    def get_snapshot(self, duration_seconds: float = 1.0) -> Dict[str, Any]:
        """
        Get analyzed snapshot of recent poses
        
        Args:
            duration_seconds: Duration to analyze (default: 1 second)
            
        Returns:
            Dictionary with aggregated pose analysis
        """
        if len(self.buffer) == 0:
            return {
                'angles': {},
                'errors': [],
                'stability': 0.0,
                'completion_percentage': 0.0,
                'frame_count': 0
            }
        
        # Calculate number of frames for duration (assuming 30 FPS)
        num_frames = int(duration_seconds * 30)
        recent_frames = self.get_last_n_frames(num_frames)
        
        if len(recent_frames) == 0:
            return {
                'angles': {},
                'errors': [],
                'stability': 0.0,
                'completion_percentage': 0.0,
                'frame_count': 0
            }
        
        # Extract current angles from latest frame
        latest = recent_frames[-1]
        current_angles = latest.get('joints', {})
        
        # Collect all errors from recent frames
        all_errors = []
        for frame in recent_frames:
            if 'errors' in frame and frame['errors']:
                all_errors.extend(frame['errors'])
        
        # Calculate stability score
        stability = self.analyze_stability(recent_frames)
        
        # Calculate completion percentage (based on balance and posture)
        completion = self._calculate_completion(recent_frames)
        
        return {
            'angles': current_angles,
            'errors': all_errors,
            'stability': stability,
            'completion_percentage': completion,
            'frame_count': len(recent_frames),
            'balance_score': latest.get('balance', {}).get('balance_score', 0.0),
            'posture_status': latest.get('posture', {}).get('status', 'Unknown')
        }
    
    def analyze_stability(self, frames: Optional[List[Dict[str, Any]]] = None) -> float:
        """
        Calculate pose stability based on variance over time
        
        Args:
            frames: List of frames to analyze (default: all frames in buffer)
            
        Returns:
            Stability score (0-100, higher is more stable)
        """
        if frames is None:
            frames = list(self.buffer)
        
        if len(frames) < 2:
            return 0.0
        
        # Extract center of gravity positions
        cog_positions = []
        for frame in frames:
            balance = frame.get('balance', {})
            cog = balance.get('cog', [0, 0])
            if cog and len(cog) == 2:
                cog_positions.append(cog)
        
        if len(cog_positions) < 2:
            return 0.0
        
        # Calculate variance in x and y positions
        cog_array = np.array(cog_positions)
        variance_x = np.var(cog_array[:, 0])
        variance_y = np.var(cog_array[:, 1])
        
        # Total variance (lower is more stable)
        total_variance = variance_x + variance_y
        
        # Convert to stability score (0-100)
        # Assuming variance < 100 is very stable, > 1000 is very unstable
        if total_variance < 10:
            stability = 100.0
        elif total_variance > 1000:
            stability = 0.0
        else:
            # Logarithmic scale for better distribution
            stability = max(0, 100 - (np.log10(total_variance) * 20))
        
        return float(stability)
    
    def _calculate_completion(self, frames: List[Dict[str, Any]]) -> float:
        """
        Calculate pose completion percentage
        
        Args:
            frames: List of frames to analyze
            
        Returns:
            Completion percentage (0-100)
        """
        if len(frames) == 0:
            return 0.0
        
        latest = frames[-1]
        
        # Base completion on balance score and posture
        balance_score = latest.get('balance', {}).get('balance_score', 0.0)
        posture_status = latest.get('posture', {}).get('status', 'Unknown')
        
        # Convert posture status to score
        posture_score = {
            'Excellent': 100,
            'Good': 80,
            'Fair': 60,
            'Poor': 40,
            'Unknown': 0
        }.get(posture_status, 0)
        
        # Weighted average (60% balance, 40% posture)
        completion = (balance_score * 0.6) + (posture_score * 0.4)
        
        return float(completion)
    
    def get_error_history(self, max_errors: int = 10) -> List[Dict[str, Any]]:
        """
        Get recent error history
        
        Args:
            max_errors: Maximum number of errors to return
            
        Returns:
            List of recent errors
        """
        all_errors = []
        
        # Collect errors from all frames (newest first)
        for frame in reversed(self.buffer):
            if 'errors' in frame and frame['errors']:
                for error in frame['errors']:
                    all_errors.append({
                        'timestamp': frame.get('timestamp', 0),
                        'frame_num': frame.get('frame_num', 0),
                        'error': error
                    })
                    
                    if len(all_errors) >= max_errors:
                        return all_errors
        
        return all_errors
    
    def clear(self) -> None:
        """Clear all data from buffer"""
        self.buffer.clear()
    
    def __len__(self) -> int:
        """Get current buffer size"""
        return len(self.buffer)
    
    def is_empty(self) -> bool:
        """Check if buffer is empty"""
        return len(self.buffer) == 0
    
    def is_full(self) -> bool:
        """Check if buffer is at max capacity"""
        return len(self.buffer) >= self.max_size
