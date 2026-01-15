"""
TTS WebSocket Client
Connects to SuperTTS and streams audio
"""

import json
import logging
import asyncio
import websockets
from typing import Optional
from collections import deque

logger = logging.getLogger(__name__)


class TTSClient:
    """WebSocket client for SuperTTS integration"""
    
    TTS_ENDPOINT = "wss://supertts.dextora.org/ws/tts"
    
    def __init__(self):
        self.ws: Optional[websockets.WebSocketClientProtocol] = None
        self.connected = False
        self.audio_queue = asyncio.Queue()
        self.reconnect_task = None
        self.listener_task = None
        
    async def connect(self):
        """Connect to SuperTTS WebSocket"""
        try:
            logger.info(f"🔊 Connecting to SuperTTS at {self.TTS_ENDPOINT}...")
            
            self.ws = await websockets.connect(
                self.TTS_ENDPOINT,
                ping_interval=30,
                ping_timeout=10
            )
            
            self.connected = True
            logger.info("✅ SuperTTS connected")
            
            # Start listener task
            self.listener_task = asyncio.create_task(self._listen_for_audio())
            
        except Exception as e:
            logger.error(f"❌ Failed to connect to SuperTTS: {e}")
            self.connected = False
            
            # Schedule reconnect
            if not self.reconnect_task or self.reconnect_task.done():
                self.reconnect_task = asyncio.create_task(self._reconnect())
    
    async def disconnect(self):
        """Close TTS connection"""
        if self.listener_task:
            self.listener_task.cancel()
        
        if self.ws:
            await self.ws.close()
        
        self.connected = False
        logger.info("👋 SuperTTS disconnected")
    
    def is_connected(self) -> bool:
        """Check if connected to TTS service"""
        return self.connected and self.ws is not None
    
    async def speak(self, text: str, language: str = "en"):
        """
        Send text to TTS and queue audio response
        
        Args:
            text: Text to convert to speech
            language: Language code (default: "en")
        """
        if not self.is_connected():
            logger.warning("⚠️ TTS not connected, skipping speech")
            return
        
        try:
            message = {
                "text": text,
                "language": language
            }
            
            logger.info(f"🗣️ Speaking: {text[:50]}...")
            
            await self.ws.send(json.dumps(message))
            
        except Exception as e:
            logger.error(f"❌ Error sending to TTS: {e}")
            await self.connect()  # Try to reconnect
    
    async def _listen_for_audio(self):
        """
        Listen for audio chunks from TTS WebSocket
        Runs as background task
        """
        try:
            async for message in self.ws:
                if isinstance(message, bytes):
                    # Audio chunk received
                    await self.audio_queue.put(message)
                    logger.debug(f"🔊 Audio chunk received: {len(message)} bytes")
                else:
                    # Text message (metadata or status)
                    logger.debug(f"📝 TTS metadata: {message}")
                    
        except websockets.exceptions.ConnectionClosed:
            logger.warning("⚠️ TTS connection closed")
            self.connected = False
            await self.connect()
        except Exception as e:
            logger.error(f"❌ Error listening to TTS: {e}")
            self.connected = False
    
    async def _reconnect(self):
        """Attempt to reconnect with exponential backoff"""
        retry_delays = [1, 2, 5, 10, 30]  # seconds
        
        for i, delay in enumerate(retry_delays):
            logger.info(f"🔄 Reconnecting to TTS (attempt {i+1}/{len(retry_delays)})...")
            
            await asyncio.sleep(delay)
            
            try:
                await self.connect()
                if self.connected:
                    logger.info("✅ TTS reconnected successfully")
                    return
            except Exception as e:
                logger.error(f"❌ Reconnect attempt {i+1} failed: {e}")
        
        logger.error("❌ All TTS reconnect attempts failed")
    
    def has_audio(self) -> bool:
        """Check if audio chunks are available"""
        return not self.audio_queue.empty()
    
    async def get_audio_chunk(self) -> bytes:
        """
        Get next audio chunk from queue
        
        Returns:
            Audio chunk as bytes
        """
        try:
            chunk = await asyncio.wait_for(self.audio_queue.get(), timeout=0.1)
            return chunk
        except asyncio.TimeoutError:
            return b""