"""
Visualization Module
Drawing utilities for frame visualization
"""

import cv2


def draw_skeleton(frame, points, pose_pairs):
    """Draw skeleton on frame"""
    for pair in pose_pairs:
        pt_A, pt_B = points[pair[0]], points[pair[1]]
        if pt_A is not None and pt_B is not None:
            cv2.line(frame, (int(pt_A[0]), int(pt_A[1])), 
                    (int(pt_B[0]), int(pt_B[1])), (0, 255, 0), 3)
            cv2.circle(frame, (int(pt_A[0]), int(pt_A[1])), 5, (0, 0, 255), -1)
            cv2.circle(frame, (int(pt_B[0]), int(pt_B[1])), 5, (0, 0, 255), -1)
    
    return frame


def draw_info_panel(frame, posture, movement, emotion):
    """Draw information panel on frame"""
    y_pos = 30
    
    # Posture info
    if posture:
        cv2.putText(frame, f"Posture: {posture['status']}", (10, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, posture['color'], 2)
        y_pos += 30
    
    # Movement info
    cv2.putText(frame, f"Energy: {movement['energy']}", (10, y_pos),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, movement['color'], 2)
    y_pos += 30
    
    # Emotion info
    cv2.putText(frame, f"Emotion: {emotion['emotion']} ({emotion['confidence']}%)", 
               (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, emotion['color'], 2)
    
    return frame
