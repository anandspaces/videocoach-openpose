"""
Gemini AI Client - Using google-genai SDK (Recommended)
Modern implementation with the latest Google Gen AI SDK
"""

import json
import logging
import asyncio
import os
from typing import Dict, Any
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)


class GeminiClient:
    """Real Gemini AI integration using google-genai (modern SDK)"""
    
    def __init__(self):
        self.connected = False
        self.client = None
        self.api_key = os.getenv("GEMINI_API_KEY")
        
    async def connect(self):
        """Initialize Gemini AI client"""
        try:
            if not self.api_key:
                logger.error("❌ GEMINI_API_KEY not found in environment")
                logger.info("💡 Add to .env file: GEMINI_API_KEY=your_key_here")
                logger.info("💡 Get key from: https://aistudio.google.com/app/apikey")
                self.connected = False
                return
            
            logger.info("🤖 Initializing Gemini AI with google-genai SDK...")
            
            # Initialize the modern Gen AI client
            self.client = genai.Client(api_key=self.api_key)
            
            self.connected = True
            logger.info("✅ Gemini AI ready (google-genai SDK)")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize Gemini: {e}")
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
        if not self.is_connected():
            logger.warning("⚠️ Gemini not connected, using fallback")
            return self._fallback_coaching(context)
        
        try:
            prompt = self._build_prompt(context)
            
            # Get response from Gemini
            response = await self._get_gemini_response(prompt)
            
            return response
            
        except Exception as e:
            logger.error(f"❌ Error getting Gemini response: {e}")
            return self._fallback_coaching(context)
    
    def _build_prompt(self, context: Dict[str, Any]) -> str:
        """Build structured prompt for Gemini"""
        posture = context.get("posture", {})
        movement = context.get("movement", {})
        emotion = context.get("emotion", {})
        balance = context.get("balance", {})
        symmetry = context.get("symmetry", {})
        issue = context.get("issue", "")
        
        prompt = f"""You are a real-time fitness coach providing live feedback during exercise.

Current User State:
- Posture Status: {posture.get('status', 'Unknown')}
- Spine Angle: {posture.get('angle', 0):.1f}° from vertical
- Movement Energy: {movement.get('energy', 'Unknown')}
- Movement Velocity: {movement.get('velocity', 0):.2f} px/frame
- Emotion: {emotion.get('emotion', 'Unknown')} (Confidence: {emotion.get('confidence', 0)}%)
- Emotional State: {emotion.get('sentiment', 'Unknown')}
- Balance Score: {balance.get('balance_score', 0):.1f}/100
- Arm Symmetry: {symmetry.get('arm_symmetry', 0):.1f}% difference
- Leg Symmetry: {symmetry.get('leg_symmetry', 0):.1f}% difference
- Detected Issue: {issue}

Provide SHORT, encouraging real-time coaching advice (1-2 sentences, under 20 words).

Requirements:
1. Be encouraging and positive
2. Give ONE specific actionable tip
3. Keep it conversational (this will be spoken aloud)
4. Focus on the detected issue: {issue}

Response:"""

        return prompt
    
    async def _get_gemini_response(self, prompt: str) -> str:
        """
        Get real response from Gemini API using google-genai SDK
        """
        try:
            # Run in thread pool since the SDK might have sync calls
            loop = asyncio.get_event_loop()
            
            # Generate response using the modern SDK
            response = await loop.run_in_executor(
                None,
                lambda: self.client.models.generate_content(
                    model='gemini-2.0-flash-exp',  # Latest fast model
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        temperature=0.7,
                        top_p=0.8,
                        top_k=40,
                        max_output_tokens=100,
                        response_modalities=["TEXT"],
                    )
                )
            )
            
            # Extract text from response
            coaching_text = response.text.strip()
            
            # Remove any quotes if present
            coaching_text = coaching_text.strip('"\'')
            
            # Ensure it's concise
            if len(coaching_text) > 150:
                # Take first sentence if too long
                coaching_text = coaching_text.split('.')[0] + '.'
            
            logger.info(f"🤖 Gemini: {coaching_text}")
            
            return coaching_text
            
        except Exception as e:
            logger.error(f"❌ Gemini API error: {e}")
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