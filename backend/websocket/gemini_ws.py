"""
Gemini AI Client - Using google-genai SDK (Recommended)
Modern implementation with the latest Google Gen AI SDK
"""

import json
import logging
import asyncio
import os
import traceback
from typing import Dict, Any
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

GEMINI_MODEL = "gemini-3-pro-preview"

# Gemini 3 Pro uses "thoughts" tokens for reasoning, so we need more output tokens
# Usage shows: ~285 prompt + ~997 thoughts + output = need at least 1500+ total
MAX_OUTPUT_TOKENS = 2048  # Increased to accommodate thoughts + actual output

class GeminiClient:
    """Real Gemini AI integration using google-genai (modern SDK)"""
    
    def __init__(self):
        self.connected = False
        self.client = None
        self.api_key = os.getenv("GEMINI_API_KEY")
        
    async def connect(self):
        """Initialize Gemini AI client"""
        logger.debug("🔧 [CONNECT] Starting Gemini client initialization...")
        try:
            if not self.api_key:
                logger.error("❌ [CONNECT] GEMINI_API_KEY not found in environment")
                logger.info("💡 Add to .env file: GEMINI_API_KEY=your_key_here")
                logger.info("💡 Get key from: https://aistudio.google.com/app/apikey")
                self.connected = False
                logger.debug("🔧 [CONNECT] Connection failed: No API key")
                return
            
            logger.info("🤖 [CONNECT] Initializing Gemini AI with google-genai SDK...")
            logger.debug(f"🔧 [CONNECT] API Key present: {self.api_key[:10]}...{self.api_key[-4:]}")
            
            # Initialize the modern Gen AI client
            self.client = genai.Client(api_key=self.api_key)
            logger.debug(f"🔧 [CONNECT] Client object created: {type(self.client)}")
            
            self.connected = True
            logger.info("✅ [CONNECT] Gemini AI ready (google-genai SDK)")
            logger.debug(f"🔧 [CONNECT] Connection status: {self.connected}")
            
        except Exception as e:
            logger.error(f"❌ [CONNECT] Failed to initialize Gemini: {e}")
            logger.error(f"❌ [CONNECT] Error type: {type(e).__name__}")
            logger.error(f"❌ [CONNECT] Traceback: {traceback.format_exc()}")
            logger.error("💡 Make sure you have: pip install google-genai")
            self.connected = False
    
    async def disconnect(self):
        """Cleanup"""
        self.connected = False
        self.client = None
        logger.info("👋 Gemini disconnected")
    
    def is_connected(self) -> bool:
        """Check if Gemini is ready"""
        return self.connected and self.client is not None
    
    async def send_coaching_request(self, context: Dict[str, Any]) -> str:
        """
        Send coaching context to Gemini and get response
        
        Args:
            context: Dictionary with pose analysis and user state
            
        Returns:
            Coaching feedback text
        """
        logger.debug(f"🔧 [COACHING_REQUEST] Starting coaching request for frame {context.get('frame_num', 'unknown')}")
        logger.debug(f"🔧 [COACHING_REQUEST] Context keys: {list(context.keys())}")
        
        if not self.is_connected():
            logger.warning("⚠️ [COACHING_REQUEST] Gemini not connected, using fallback")
            logger.debug(f"🔧 [COACHING_REQUEST] Connection status: connected={self.connected}, client={self.client is not None}")
            return self._fallback_coaching(context)
        
        logger.debug("🔧 [COACHING_REQUEST] Gemini is connected, proceeding with request")
        
        try:
            logger.debug("🔧 [COACHING_REQUEST] Building prompt from context...")
            prompt = self._build_prompt(context)
            logger.debug(f"🔧 [COACHING_REQUEST] Prompt built, length: {len(prompt)} characters")
            
            # Get response from Gemini
            logger.debug("🔧 [COACHING_REQUEST] Sending request to Gemini API...")
            response = await self._get_gemini_response(prompt)
            logger.debug(f"🔧 [COACHING_REQUEST] Response received: {response[:100] if response else 'None'}...")
            
            return response
            
        except Exception as e:
            logger.error(f"❌ [COACHING_REQUEST] Error getting Gemini response: {e}")
            logger.error(f"❌ [COACHING_REQUEST] Error type: {type(e).__name__}")
            logger.error(f"❌ [COACHING_REQUEST] Context keys: {list(context.keys())}")
            logger.error(f"❌ [COACHING_REQUEST] Traceback: {traceback.format_exc()}")
            return self._fallback_coaching(context)
    
    def _build_prompt(self, context: Dict[str, Any]) -> str:
        """Build structured prompt for Gemini with actual movement data"""
        logger.debug(f"🔧 [BUILD_PROMPT] Building prompt for frame {context.get('frame_num', 0)}")
        
        posture = context.get("posture", {})
        movement = context.get("movement", {})
        emotion = context.get("emotion", {})
        balance = context.get("balance", {})
        symmetry = context.get("symmetry", {})
        joints = context.get("joints", {})
        keypoints = context.get("keypoints", {})
        frame_num = context.get("frame_num", 0)
        
        logger.debug(f"🔧 [BUILD_PROMPT] Context data extracted:")
        logger.debug(f"  - Posture: {posture}")
        logger.debug(f"  - Movement: {movement}")
        logger.debug(f"  - Emotion: {emotion}")
        logger.debug(f"  - Balance: {balance}")
        logger.debug(f"  - Symmetry: {symmetry}")
        logger.debug(f"  - Joints count: {len(joints)}")
        logger.debug(f"  - Keypoints count: {len(keypoints)}")
        
        # Build detailed movement context
        movement_energy = movement.get('energy', 'Unknown')
        movement_score = movement.get('movement_score', 0)
        velocity = movement.get('velocity', 0)
        
        balance_score = balance.get('balance_score', 50)
        cog = balance.get('center_of_gravity', {})
        
        arm_symmetry = symmetry.get('arm_symmetry', 0)
        leg_symmetry = symmetry.get('leg_symmetry', 0)
        
        posture_status = posture.get('status', 'Unknown')
        spine_angle = posture.get('angle', 0)
        
        emotion_state = emotion.get('emotion', 'Unknown')
        emotion_confidence = emotion.get('confidence', 0)
        
        # Build joint angles summary
        joint_info = []
        for joint_name, angle in joints.items():
            if angle > 0:  # Only include detected joints
                state = "BENT" if angle < 140 else "EXTENDED"
                joint_info.append(f"{joint_name}: {angle:.0f}° [{state}]")
        
        joints_str = ", ".join(joint_info[:5]) if joint_info else "No clear joint angles detected"
        
        # Build keypoint positions summary (only key points)
        key_positions = []
        important_points = ['Nose', 'Neck', 'RShoulder', 'LShoulder', 'RHip', 'LHip', 
                          'RElbow', 'LElbow', 'RKnee', 'LKnee']
        
        logger.debug(f"🔧 [BUILD_PROMPT] Keypoints type: {type(keypoints)}")
        logger.debug(f"🔧 [BUILD_PROMPT] Keypoints keys: {list(keypoints.keys()) if isinstance(keypoints, dict) else 'Not a dict'}")
        
        for point_name in important_points:
            if point_name in keypoints:
                kp = keypoints[point_name]
                logger.debug(f"🔧 [BUILD_PROMPT] Processing {point_name}: type={type(kp)}, value={kp}")
                
                try:
                    # Handle dict format (current format from main.py)
                    if isinstance(kp, dict):
                        x = kp.get('x', 0)
                        y = kp.get('y', 0)
                        conf = kp.get('confidence', 0)
                        logger.debug(f"🔧 [BUILD_PROMPT] {point_name} dict format: x={x:.1f}, y={y:.1f}, conf={conf:.2f}")
                        if conf > 0.2:  # confidence threshold
                            key_positions.append(f"{point_name}:({x:.0f},{y:.0f})")
                            logger.debug(f"🔧 [BUILD_PROMPT] Added {point_name} to positions (conf={conf:.2f})")
                    # Handle tuple/list format (legacy)
                    elif isinstance(kp, (tuple, list)) and len(kp) >= 3:
                        x, y, conf = kp[0], kp[1], kp[2]
                        logger.debug(f"🔧 [BUILD_PROMPT] {point_name} tuple/list format: x={x:.1f}, y={y:.1f}, conf={conf:.2f}")
                        if conf > 0.2:  # confidence threshold
                            key_positions.append(f"{point_name}:({x:.0f},{y:.0f})")
                            logger.debug(f"🔧 [BUILD_PROMPT] Added {point_name} to positions (conf={conf:.2f})")
                    else:
                        logger.warning(f"⚠️ [BUILD_PROMPT] Unexpected keypoint format for {point_name}: {type(kp)}")
                except Exception as e:
                    logger.error(f"❌ [BUILD_PROMPT] Error processing keypoint {point_name}: {e}")
                    logger.error(f"❌ [BUILD_PROMPT] Traceback: {traceback.format_exc()}")
        
        positions_str = ", ".join(key_positions[:6]) if key_positions else "Limited keypoints detected"
        
        logger.debug(f"🔧 [BUILD_PROMPT] Positions string: {positions_str}")
        logger.debug(f"🔧 [BUILD_PROMPT] Joints string: {joints_str}")
        
        # Simplified prompt to reduce token usage (Gemini 3 Pro uses lots of tokens for "thoughts")
        prompt = f"""Fitness coach analyzing pose data. Frame {frame_num}.

Body positions: {positions_str}
Joints: {joints_str}
Energy: {movement_energy}, Balance: {balance_score:.0f}/100
Emotion: {emotion_state}

Give ONE specific 10-word coaching tip based on the positions above:"""
        
        logger.debug(f"🔧 [BUILD_PROMPT] Complete prompt:\n{prompt}")
        logger.info(f"📝 [BUILD_PROMPT] Prompt built successfully with {len(key_positions)} keypoints and {len(joint_info)} joints")
        
        return prompt

    
    async def _get_gemini_response(self, prompt: str) -> str:
        """
        Get real response from Gemini API using google-genai SDK
        """
        logger.debug(f"🔧 [GEMINI_API] Starting API request to {GEMINI_MODEL}")
        logger.debug(f"🔧 [GEMINI_API] Prompt length: {len(prompt)} characters")
        
        try:
            # Run in thread pool since the SDK might have sync calls
            loop = asyncio.get_event_loop()
            
            logger.debug("🔧 [GEMINI_API] Preparing API configuration...")
            config = types.GenerateContentConfig(
                temperature=0.7,  # Reduced for more focused responses
                top_p=0.9,
                top_k=20,
                max_output_tokens=MAX_OUTPUT_TOKENS,  # Increased for thoughts + output
                response_modalities=["TEXT"],
            )
            logger.debug(f"🔧 [GEMINI_API] Config: temp={config.temperature}, top_p={config.top_p}, max_tokens={config.max_output_tokens}")
            
            # Generate response using the modern SDK
            logger.info(f"🌐 [GEMINI_API] Sending request to Gemini API...")
            response = await loop.run_in_executor(
                None,
                lambda: self.client.models.generate_content(
                    model=GEMINI_MODEL,
                    contents=prompt,
                    config=config
                )
            )
            
            logger.debug(f"🔧 [GEMINI_API] Response received, type: {type(response)}")
            logger.debug(f"🔧 [GEMINI_API] Response attributes: {dir(response)}")
            
            # Log response structure for debugging
            logger.debug(f"🔧 [GEMINI_API] Response object inspection:")
            logger.debug(f"  - Has 'text' attr: {hasattr(response, 'text')}")
            logger.debug(f"  - Has 'candidates' attr: {hasattr(response, 'candidates')}")
            logger.debug(f"  - Has 'parts' attr: {hasattr(response, 'parts')}")
            
            if hasattr(response, 'candidates'):
                logger.debug(f"  - Candidates count: {len(response.candidates) if response.candidates else 0}")
                if response.candidates:
                    logger.debug(f"  - First candidate type: {type(response.candidates[0])}")
                    logger.debug(f"  - First candidate attrs: {dir(response.candidates[0])}")
            
            # Extract text from response
            # Note: response.text works on gemini-3-pro-preview once tokens are sufficient
            coaching_text = None
            extraction_method = None
            
            # Method 1: Direct text attribute
            if hasattr(response, 'text') and response.text:
                coaching_text = response.text.strip()
                extraction_method = "direct_text"
                logger.debug(f"🔧 [GEMINI_API] Extracted via direct text: {coaching_text[:100]}...")
            
            # Method 2: Candidates structure
            elif hasattr(response, 'candidates') and response.candidates:
                logger.debug("🔧 [GEMINI_API] Attempting extraction from candidates...")
                try:
                    candidate = response.candidates[0]
                    logger.debug(f"🔧 [GEMINI_API] Candidate has content: {hasattr(candidate, 'content')}")
                    
                    # Check finish reason - if MAX_TOKENS, the model ran out of space
                    if hasattr(candidate, 'finish_reason'):
                        finish_reason = str(candidate.finish_reason)
                        logger.debug(f"🔧 [GEMINI_API] Finish reason: {finish_reason}")
                        if 'MAX_TOKENS' in finish_reason:
                            logger.error("❌ [GEMINI_API] Response hit MAX_TOKENS limit - model used all tokens for thoughts!")
                            logger.error(f"❌ [GEMINI_API] Usage: {response.usage_metadata if hasattr(response, 'usage_metadata') else 'N/A'}")
                    
                    if hasattr(candidate, 'content') and candidate.content:
                        logger.debug(f"🔧 [GEMINI_API] Content type: {type(candidate.content)}")
                        logger.debug(f"🔧 [GEMINI_API] Content has parts: {hasattr(candidate.content, 'parts')}")
                        
                        if hasattr(candidate.content, 'parts') and candidate.content.parts:
                            logger.debug(f"🔧 [GEMINI_API] Parts count: {len(candidate.content.parts)}")
                            if len(candidate.content.parts) > 0:
                                part = candidate.content.parts[0]
                                logger.debug(f"🔧 [GEMINI_API] Part type: {type(part)}")
                                logger.debug(f"🔧 [GEMINI_API] Part has text: {hasattr(part, 'text')}")
                                
                                if hasattr(part, 'text') and part.text:
                                    coaching_text = part.text.strip()
                                    extraction_method = "candidates_parts"
                                    logger.debug(f"🔧 [GEMINI_API] Extracted via candidates.parts: {coaching_text[:100]}...")
                                else:
                                    logger.warning("⚠️ [GEMINI_API] Part has no text attribute or text is empty")
                            else:
                                logger.warning("⚠️ [GEMINI_API] Parts list is empty")
                        else:
                            logger.warning("⚠️ [GEMINI_API] Content has no parts or parts is None")
                except (IndexError, AttributeError) as e:
                    logger.error(f"❌ [GEMINI_API] Error extracting from candidates: {e}")
                    logger.error(f"❌ [GEMINI_API] Traceback: {traceback.format_exc()}")
            
            # Check if we got nothing due to MAX_TOKENS
            if not coaching_text:
                if hasattr(response, 'candidates') and response.candidates:
                    candidate = response.candidates[0]
                    if hasattr(candidate, 'finish_reason') and 'MAX_TOKENS' in str(candidate.finish_reason):
                        logger.error("❌ [GEMINI_API] MAX_TOKENS error - increasing max_output_tokens needed")
                        raise ValueError("Gemini hit MAX_TOKENS limit - all tokens used for internal reasoning")
                
                logger.error("❌ [GEMINI_API] Could not extract text from Gemini response")
                logger.error(f"❌ [GEMINI_API] Response dump: {response}")
                raise ValueError("Empty Gemini response")
            
            logger.info(f"✅ [GEMINI_API] Text extracted successfully via {extraction_method}")
            
            # Remove any quotes if present
            coaching_text = coaching_text.strip('"\'')
            logger.debug(f"🔧 [GEMINI_API] After quote removal: {coaching_text}")
            
            # Ensure it's concise
            if len(coaching_text) > 150:
                original_length = len(coaching_text)
                # Take first sentence if too long
                coaching_text = coaching_text.split('.')[0] + '.'
                logger.debug(f"🔧 [GEMINI_API] Truncated from {original_length} to {len(coaching_text)} chars")
            
            logger.info(f"🤖 [GEMINI_API] Final coaching text: {coaching_text}")
            
            return coaching_text
            
        except Exception as e:
            logger.error(f"❌ [GEMINI_API] API error: {e}")
            logger.error(f"❌ [GEMINI_API] Error type: {type(e).__name__}")
            logger.error(f"❌ [GEMINI_API] Traceback: {traceback.format_exc()}")
            raise
    
    def _fallback_coaching(self, context: Dict[str, Any]) -> str:
        """
        Generate fallback coaching when Gemini unavailable
        """
        issue = context.get("issue", "")
        
        fallback_map = {
            "poor_balance": "Focus on centering your weight. Engage your core for stability.",
            "poor_posture": "Stand taller! Imagine a string pulling you up from the crown of your head.",
            "asymmetry": "Keep your body balanced. Check your shoulder and hip alignment.",
            "low_confidence": "You're doing great! Keep moving with confidence.",
            "high_energy": "Great energy! Make sure your movements are controlled.",
            "low_energy": "Let's pick up the pace! Add some energy to your movements.",
            "frustration": "Take a breath. You're making progress, stay focused.",
        }
        
        return fallback_map.get(issue, "Keep up the great work! Stay focused on your form.")