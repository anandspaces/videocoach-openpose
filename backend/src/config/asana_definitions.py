"""
Asana Definitions Database
Contains ideal alignments, detection rules, and common mistakes for yoga asanas
"""

from typing import Dict, List, Any

ASANA_DEFINITIONS: Dict[str, Dict[str, Any]] = {
    "tree_pose": {
        "name": "Tree Pose (Vrksasana)",
        "description": "A standing balance pose with one foot on the inner thigh",
        
        "detection_rules": {
            # One leg should be straight (standing leg)
            "has_straight_leg": lambda joints: any(angle > 160 for angle in [joints.get('RKnee', 0), joints.get('LKnee', 0)]),
            # One leg should be bent (raised leg)
            "has_bent_leg": lambda joints: any(angle < 100 for angle in [joints.get('RKnee', 0), joints.get('LKnee', 0)]),
            # Hips should be relatively level
            "hips_level": lambda keypoints: abs(keypoints.get('RHip', {}).get('y', 0) - keypoints.get('LHip', {}).get('y', 0)) < 40,
            # Balance score should be moderate to high
            "balance_required": lambda balance: balance.get('balance_score', 0) > 35,
        },
        
        "ideal_alignment": [
            "Standing leg completely straight and engaged",
            "Raised foot placed on inner thigh, NOT on knee joint",
            "Hips square and facing forward",
            "Core engaged to maintain balance",
            "Shoulders relaxed, chest open",
            "Gaze steady at a fixed point (drishti)"
        ],
        
        "common_mistakes": [
            "Placing raised foot directly on the knee joint (can cause injury)",
            "Hips tilting to one side instead of staying square",
            "Standing knee hyperextended or locked",
            "Shoulders hunched up toward ears",
            "Looking down instead of maintaining steady gaze",
            "Leaning to one side instead of staying centered"
        ],
        
        "key_corrections": {
            "foot_on_knee": "Move your raised foot to your inner thigh or calf, never on the knee joint.",
            "tilted_hips": "Draw your raised leg's hip back to square your hips forward.",
            "bent_standing_leg": "Engage your standing leg muscles and straighten your knee completely.",
            "hunched_shoulders": "Relax your shoulders down away from your ears.",
            "poor_balance": "Fix your gaze on a steady point ahead and engage your core."
        }
    },
    
    "mountain_pose": {
        "name": "Mountain Pose (Tadasana)",
        "description": "A foundational standing pose with feet together and body aligned",
        
        "detection_rules": {
            # Both legs should be straight
            "legs_straight": lambda joints: joints.get('RKnee', 0) > 160 and joints.get('LKnee', 0) > 160,
            # Feet should be close together
            "feet_together": lambda keypoints: abs(keypoints.get('RAnkle', {}).get('x', 0) - keypoints.get('LAnkle', {}).get('x', 0)) < 50,
            # Upright posture
            "upright": lambda posture: posture.get('angle', 0) > 80,
            # Arms likely at sides or overhead
            "arms_position": lambda joints: True,  # Flexible for arms
        },
        
        "ideal_alignment": [
            "Feet together or hip-width apart, parallel",
            "Weight evenly distributed across both feet",
            "Legs straight and engaged, kneecaps lifted",
            "Pelvis neutral, tailbone slightly tucked",
            "Spine elongated, crown of head reaching up",
            "Shoulders back and down, chest open",
            "Arms at sides with palms facing forward, or overhead"
        ],
        
        "common_mistakes": [
            "Weight shifted to one leg instead of balanced",
            "Knees locked or hyperextended",
            "Pelvis tilted forward (anterior pelvic tilt)",
            "Shoulders rounded forward",
            "Chin jutting forward instead of neutral",
            "Holding breath instead of breathing naturally"
        ],
        
        "key_corrections": {
            "uneven_weight": "Distribute your weight evenly across both feet, all four corners grounded.",
            "locked_knees": "Soften your knees slightly while keeping legs engaged.",
            "rounded_shoulders": "Roll your shoulders back and down, opening your chest.",
            "forward_head": "Align your ears over your shoulders, chin parallel to the floor.",
            "shallow_breathing": "Breathe deeply and naturally, allowing your ribcage to expand."
        }
    },
    
    "warrior_1": {
        "name": "Warrior 1 (Virabhadrasana I)",
        "description": "A standing lunge pose with front knee bent and arms overhead",
        
        "detection_rules": {
            # One knee bent (front leg)
            "front_knee_bent": lambda joints: any(angle < 130 for angle in [joints.get('RKnee', 0), joints.get('LKnee', 0)]),
            # One leg straight (back leg)
            "back_leg_straight": lambda joints: any(angle > 150 for angle in [joints.get('RKnee', 0), joints.get('LKnee', 0)]),
            # Arms likely overhead
            "arms_raised": lambda joints: any(angle > 140 for angle in [joints.get('RShoulder', 0), joints.get('LShoulder', 0)]),
            # Wide stance
            "wide_stance": lambda keypoints: abs(keypoints.get('RAnkle', {}).get('x', 0) - keypoints.get('LAnkle', {}).get('x', 0)) > 100,
        },
        
        "ideal_alignment": [
            "Front knee bent to 90 degrees, aligned over ankle",
            "Back leg straight and strong",
            "Hips square forward (both hip points facing front)",
            "Torso upright, spine elongated",
            "Arms reaching overhead, shoulder-width apart",
            "Shoulders relaxed, gaze forward or up"
        ],
        
        "common_mistakes": [
            "Front knee collapsing inward past the ankle",
            "Back foot turned out too much (should be 45-60 degrees)",
            "Hips not square - back hip open to the side",
            "Arching lower back excessively",
            "Shoulders hunched up toward ears",
            "Leaning torso forward instead of staying upright"
        ],
        
        "key_corrections": {
            "knee_over_toe": "Align your front knee directly over your ankle, not past your toes.",
            "open_hips": "Rotate your back hip forward to square your hips toward the front.",
            "collapsed_arch": "Press through the outer edge of your back foot to engage the leg.",
            "arched_back": "Engage your core and lengthen your tailbone down to protect your lower back.",
            "hunched_shoulders": "Relax your shoulders away from your ears while reaching arms up."
        }
    },
    
    "warrior_2": {
        "name": "Warrior 2 (Virabhadrasana II)",
        "description": "A standing lunge pose with arms extended to the sides",
        
        "detection_rules": {
            # One knee bent
            "front_knee_bent": lambda joints: any(angle < 130 for angle in [joints.get('RKnee', 0), joints.get('LKnee', 0)]),
            # One leg straight
            "back_leg_straight": lambda joints: any(angle > 150 for angle in [joints.get('RKnee', 0), joints.get('LKnee', 0)]),
            # Arms extended to sides (not overhead like W1)
            "arms_extended": lambda joints: joints.get('RElbow', 0) > 140 and joints.get('LElbow', 0) > 140,
            # Wide stance
            "wide_stance": lambda keypoints: abs(keypoints.get('RAnkle', {}).get('x', 0) - keypoints.get('LAnkle', {}).get('x', 0)) > 100,
        },
        
        "ideal_alignment": [
            "Front knee bent to 90 degrees, aligned over ankle",
            "Back leg straight, foot at 90 degrees",
            "Hips open to the side (not square like Warrior 1)",
            "Torso upright, centered between legs",
            "Arms extended at shoulder height, parallel to floor",
            "Gaze over front fingertips"
        ],
        
        "common_mistakes": [
            "Front knee collapsing inward",
            "Front knee not bent enough (should be 90 degrees)",
            "Leaning torso over front leg",
            "Arms drooping below shoulder height",
            "Back foot not turned out enough",
            "Shoulders hunched or tense"
        ],
        
        "key_corrections": {
            "shallow_bend": "Bend your front knee deeper until your thigh is parallel to the floor.",
            "knee_inward": "Press your front knee out toward your pinky toe to align with your ankle.",
            "leaning_forward": "Stack your torso directly over your hips, don't lean toward your front leg.",
            "drooping_arms": "Lift your arms to shoulder height and extend actively through your fingertips.",
            "back_foot_angle": "Turn your back foot out to 90 degrees for better stability."
        }
    },
    
    "downward_dog": {
        "name": "Downward-Facing Dog (Adho Mukha Svanasana)",
        "description": "An inverted V-shape pose with hands and feet on the ground",
        
        "detection_rules": {
            # Both knees should be relatively straight
            "legs_straight": lambda joints: joints.get('RKnee', 0) > 140 and joints.get('LKnee', 0) > 140,
            # Both elbows should be straight
            "arms_straight": lambda joints: joints.get('RElbow', 0) > 150 and joints.get('LElbow', 0) > 150,
            # Hips should be highest point (inverted V)
            "hips_elevated": lambda keypoints: keypoints.get('RHip', {}).get('y', 500) < keypoints.get('Neck', {}).get('y', 0),
            # Body in inverted position
            "inverted": lambda posture: posture.get('angle', 90) < 60,
        },
        
        "ideal_alignment": [
            "Hands shoulder-width apart, fingers spread wide",
            "Arms straight, shoulders externally rotated",
            "Spine long and straight, tailbone reaching up",
            "Legs straight (or knees slightly bent if hamstrings tight)",
            "Heels reaching toward the floor",
            "Head relaxed between arms, neck neutral"
        ],
        
        "common_mistakes": [
            "Shoulders hunched up toward ears",
            "Rounding the spine instead of lengthening",
            "Hands too close together or too far apart",
            "Weight shifted forward into hands instead of back into legs",
            "Knees locked or hyperextended",
            "Holding breath"
        ],
        
        "key_corrections": {
            "rounded_back": "Bend your knees slightly and focus on lengthening your spine from tailbone to crown.",
            "hunched_shoulders": "Externally rotate your shoulders and draw shoulder blades down your back.",
            "weight_forward": "Shift your weight back into your legs, pressing your hips up and back.",
            "locked_knees": "Soften your knees slightly to protect your joints and lengthen your spine.",
            "hand_position": "Place your hands shoulder-width apart with fingers spread for better stability."
        }
    }
}


def get_asana_names() -> List[str]:
    """Get list of all supported asana names"""
    return list(ASANA_DEFINITIONS.keys())


def get_asana_info(asana_name: str) -> Dict[str, Any]:
    """Get full information for a specific asana"""
    return ASANA_DEFINITIONS.get(asana_name, {})


def get_ideal_alignment(asana_name: str) -> str:
    """Get ideal alignment description for an asana"""
    asana = ASANA_DEFINITIONS.get(asana_name, {})
    alignments = asana.get('ideal_alignment', [])
    return "\n".join(f"- {alignment}" for alignment in alignments)


def get_common_mistakes(asana_name: str) -> str:
    """Get common mistakes for an asana"""
    asana = ASANA_DEFINITIONS.get(asana_name, {})
    mistakes = asana.get('common_mistakes', [])
    return "\n".join(f"- {mistake}" for mistake in mistakes)


def get_key_corrections(asana_name: str) -> Dict[str, str]:
    """Get key corrections dictionary for an asana"""
    asana = ASANA_DEFINITIONS.get(asana_name, {})
    return asana.get('key_corrections', {})
