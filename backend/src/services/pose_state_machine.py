"""
Pose State Machine
Temporal state tracking for yoga poses

States:
- INIT: No pose detected or starting state
- ENTERING_POSE: User transitioning into a pose
- POSE_HOLD: User holding a stable pose (ONLY state where feedback is given)
- TRANSITION: Moving between poses
- EXIT: Leaving current pose

Critical rules:
1. State transitions are TIME-GATED (no instant flipping)
2. POSE_HOLD requires minimum stability duration
3. TRANSITION state IGNORES alignment errors
4. Hysteresis prevents oscillation
"""

from enum import Enum
from typing import Dict, Optional, Tuple
from collections import deque
import time
import numpy as np


class PoseState(Enum):
    """Pose phase states"""
    INIT = "INIT"
    ENTERING_POSE = "ENTERING_POSE"
    POSE_HOLD = "POSE_HOLD"
    TRANSITION = "TRANSITION"
    EXIT = "EXIT"


class MotionBuffer:
    """
    Rolling buffer for temporal motion analysis
    Tracks joint angles over time to compute velocity and stability
    """
    
    def __init__(self, max_frames: int = 60):
        """
        Args:
            max_frames: Maximum number of frames to keep (default 60 = 2 seconds at 30fps)
        """
        self.max_frames = max_frames
        self.angle_history: Dict[str, deque] = {}
        self.timestamp_history = deque(maxlen=max_frames)
    
    def add_frame(self, joint_angles: Dict[str, float], timestamp: float):
        """
        Add a frame of joint angles
        
        Args:
            joint_angles: Dictionary of joint_name -> angle in degrees
            timestamp: Frame timestamp in seconds
        """
        self.timestamp_history.append(timestamp)
        
        for joint, angle in joint_angles.items():
            if joint not in self.angle_history:
                self.angle_history[joint] = deque(maxlen=self.max_frames)
            self.angle_history[joint].append(angle)
    
    def get_angular_velocity(self, joint: str, window: int = 10) -> float:
        """
        Calculate angular velocity for a joint
        
        Args:
            joint: Joint name
            window: Number of frames to analyze (default 10 = ~0.33s at 30fps)
            
        Returns:
            Angular velocity in degrees/second
        """
        if joint not in self.angle_history:
            return 0.0
        
        history = list(self.angle_history[joint])
        if len(history) < 2:
            return 0.0
        
        # Use last N frames
        recent = history[-min(window, len(history)):]
        timestamps = list(self.timestamp_history)[-len(recent):]
        
        if len(recent) < 2 or len(timestamps) < 2:
            return 0.0
        
        # Calculate velocity using linear regression
        time_diffs = np.diff(timestamps)
        angle_diffs = np.diff(recent)
        
        if len(time_diffs) == 0 or np.sum(time_diffs) == 0:
            return 0.0
        
        # Average velocity
        velocities = angle_diffs / (time_diffs + 1e-6)
        return float(np.mean(np.abs(velocities)))
    
    def get_stability_score(self, joints: list, window: int = 30) -> float:
        """
        Calculate overall stability score (0.0 = moving, 1.0 = stable)
        
        Args:
            joints: List of joint names to check
            window: Number of frames to analyze
            
        Returns:
            Stability score between 0.0 and 1.0
        """
        velocities = []
        
        for joint in joints:
            vel = self.get_angular_velocity(joint, window)
            velocities.append(vel)
        
        if not velocities:
            return 0.0
        
        # Average velocity
        avg_velocity = np.mean(velocities)
        
        # Convert to stability (high velocity = low stability)
        # Threshold: 5 degrees/second is considered stable
        stability = 1.0 / (1.0 + avg_velocity / 5.0)
        
        return float(stability)
    
    def get_angle_variance(self, joint: str, window: int = 30) -> float:
        """
        Calculate variance of joint angle over time
        
        Args:
            joint: Joint name
            window: Number of frames to analyze
            
        Returns:
            Variance in degrees²
        """
        if joint not in self.angle_history:
            return 0.0
        
        history = list(self.angle_history[joint])
        if len(history) < 2:
            return 0.0
        
        recent = history[-min(window, len(history)):]
        return float(np.var(recent))
    
    def clear(self):
        """Clear all history"""
        self.angle_history.clear()
        self.timestamp_history.clear()


class PoseStateMachine:
    """
    State machine for pose phase tracking
    
    Enforces temporal rules:
    - Time-gated transitions
    - Stability requirements
    - Hysteresis to prevent oscillation
    """
    
    # Timing constants (in seconds)
    MIN_ENTERING_DURATION = 0.5    # Minimum time in ENTERING before HOLD
    MIN_HOLD_DURATION = 1.0        # Minimum time in HOLD before allowing EXIT
    MIN_STABILITY_DURATION = 1.5   # Minimum stable time before POSE_HOLD
    TRANSITION_TIMEOUT = 3.0       # Max time in TRANSITION before EXIT
    
    # Stability thresholds
    STABILITY_THRESHOLD = 0.75     # Minimum stability score for POSE_HOLD
    VELOCITY_THRESHOLD = 10.0      # Max velocity (deg/s) for stability
    
    def __init__(self, asana_name: Optional[str] = None):
        """
        Args:
            asana_name: Name of the asana being tracked (optional)
        """
        self.asana_name = asana_name
        self.current_state = PoseState.INIT
        self.state_entry_time = time.time()
        self.motion_buffer = MotionBuffer(max_frames=60)
        
        # State history for debugging
        self.state_history = []
        
        # Hysteresis tracking
        self.consecutive_stable_frames = 0
        self.consecutive_moving_frames = 0
    
    def update(self, joint_angles: Dict[str, float], timestamp: float) -> PoseState:
        """
        Update state machine with new frame data
        
        Args:
            joint_angles: Current joint angles
            timestamp: Frame timestamp
            
        Returns:
            Current state after update
        """
        # Add to motion buffer
        self.motion_buffer.add_frame(joint_angles, timestamp)
        
        # Calculate metrics
        stability = self.motion_buffer.get_stability_score(list(joint_angles.keys()))
        time_in_state = timestamp - self.state_entry_time
        
        # State transition logic
        new_state = self._compute_next_state(stability, time_in_state, timestamp)
        
        # If state changed, record it
        if new_state != self.current_state:
            self._transition_to(new_state, timestamp)
        
        return self.current_state
    
    def _compute_next_state(self, stability: float, time_in_state: float, 
                           timestamp: float) -> PoseState:
        """
        Compute next state based on current state and metrics
        
        Args:
            stability: Current stability score (0-1)
            time_in_state: Time spent in current state (seconds)
            timestamp: Current timestamp
            
        Returns:
            Next state
        """
        current = self.current_state
        
        # Track consecutive stable/moving frames for hysteresis
        if stability >= self.STABILITY_THRESHOLD:
            self.consecutive_stable_frames += 1
            self.consecutive_moving_frames = 0
        else:
            self.consecutive_moving_frames += 1
            self.consecutive_stable_frames = 0
        
        # State transition rules
        if current == PoseState.INIT:
            # INIT → ENTERING_POSE when movement detected
            if stability < 0.9:  # Some movement
                return PoseState.ENTERING_POSE
            return PoseState.INIT
        
        elif current == PoseState.ENTERING_POSE:
            # ENTERING_POSE → POSE_HOLD when stable for minimum duration
            if (stability >= self.STABILITY_THRESHOLD and 
                time_in_state >= self.MIN_ENTERING_DURATION and
                self.consecutive_stable_frames >= 15):  # ~0.5s at 30fps
                return PoseState.POSE_HOLD
            
            # ENTERING_POSE → TRANSITION if too much movement
            if stability < 0.3 and time_in_state > 1.0:
                return PoseState.TRANSITION
            
            return PoseState.ENTERING_POSE
        
        elif current == PoseState.POSE_HOLD:
            # POSE_HOLD → TRANSITION when movement detected
            # Hysteresis: require multiple consecutive moving frames
            if (stability < self.STABILITY_THRESHOLD and 
                self.consecutive_moving_frames >= 10):  # ~0.33s at 30fps
                return PoseState.TRANSITION
            
            # POSE_HOLD → EXIT if minimum hold time met and intentional exit
            if (time_in_state >= self.MIN_HOLD_DURATION and
                stability < 0.5 and
                self.consecutive_moving_frames >= 20):  # ~0.67s at 30fps
                return PoseState.EXIT
            
            return PoseState.POSE_HOLD
        
        elif current == PoseState.TRANSITION:
            # TRANSITION → ENTERING_POSE when stabilizing
            if (stability >= self.STABILITY_THRESHOLD and
                self.consecutive_stable_frames >= 10):
                return PoseState.ENTERING_POSE
            
            # TRANSITION → EXIT if timeout
            if time_in_state >= self.TRANSITION_TIMEOUT:
                return PoseState.EXIT
            
            return PoseState.TRANSITION
        
        elif current == PoseState.EXIT:
            # EXIT → INIT (reset)
            if time_in_state >= 0.5:  # Brief pause before allowing new pose
                return PoseState.INIT
            
            return PoseState.EXIT
        
        return current
    
    def _transition_to(self, new_state: PoseState, timestamp: float):
        """
        Transition to a new state
        
        Args:
            new_state: State to transition to
            timestamp: Transition timestamp
        """
        old_state = self.current_state
        self.current_state = new_state
        self.state_entry_time = timestamp
        
        # Record transition
        self.state_history.append({
            'from': old_state.value,
            'to': new_state.value,
            'timestamp': timestamp
        })
        
        # Keep history limited
        if len(self.state_history) > 50:
            self.state_history.pop(0)
        
        # Reset hysteresis counters on state change
        self.consecutive_stable_frames = 0
        self.consecutive_moving_frames = 0
    
    def should_evaluate_alignment(self) -> bool:
        """
        Check if alignment should be evaluated in current state
        
        Returns:
            True if alignment evaluation is allowed
        """
        # ONLY evaluate alignment in POSE_HOLD state
        return self.current_state == PoseState.POSE_HOLD
    
    def get_time_in_state(self) -> float:
        """
        Get time spent in current state
        
        Returns:
            Time in seconds
        """
        return time.time() - self.state_entry_time
    
    def get_state_info(self) -> Dict:
        """
        Get current state information
        
        Returns:
            Dictionary with state details
        """
        return {
            'state': self.current_state.value,
            'asana': self.asana_name,
            'time_in_state': self.get_time_in_state(),
            'can_evaluate': self.should_evaluate_alignment(),
            'stability': self.motion_buffer.get_stability_score(
                list(self.motion_buffer.angle_history.keys())
            ) if self.motion_buffer.angle_history else 0.0
        }
    
    def reset(self):
        """Reset state machine to INIT"""
        self.current_state = PoseState.INIT
        self.state_entry_time = time.time()
        self.motion_buffer.clear()
        self.consecutive_stable_frames = 0
        self.consecutive_moving_frames = 0
    
    def set_asana(self, asana_name: str):
        """
        Set the asana being tracked
        
        Args:
            asana_name: Name of the asana
        """
        self.asana_name = asana_name
        # Don't reset state - allow changing asana mid-flow
