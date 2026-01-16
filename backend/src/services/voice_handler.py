"""
Voice Handler - Manages voice activity detection, speech-to-text, and text-to-speech
Supports both Google Cloud APIs and Web Speech API fallback
"""

import asyncio
import logging
from typing import Optional, Callable, Dict, Any
import base64
import io

logger = logging.getLogger(__name__)


class VoiceHandler:
    """
    Handles voice processing: VAD, STT, and TTS
    Uses Web Speech API (browser-side) by default for simplicity
    Can be extended to use Google Cloud Speech APIs for higher quality
    """
    
    def __init__(
        self,
        use_google_cloud: bool = False,
        google_credentials_path: Optional[str] = None
    ):
        """
        Initialize voice handler
        
        Args:
            use_google_cloud: Use Google Cloud Speech APIs (default: False, uses Web Speech API)
            google_credentials_path: Path to Google Cloud credentials JSON
        """
        self.use_google_cloud = use_google_cloud
        self.google_credentials_path = google_credentials_path
        
        self.stt_client = None
        self.tts_client = None
        
        if use_google_cloud:
            self._init_google_cloud()
    
    def _init_google_cloud(self) -> None:
        """Initialize Google Cloud Speech clients"""
        try:
            from google.cloud import speech_v1 as speech
            from google.cloud import texttospeech
            
            # Initialize STT client
            self.stt_client = speech.SpeechClient()
            logger.info("✅ Google Cloud Speech-to-Text initialized")
            
            # Initialize TTS client
            self.tts_client = texttospeech.TextToSpeechClient()
            logger.info("✅ Google Cloud Text-to-Speech initialized")
            
        except ImportError:
            logger.warning("⚠️ Google Cloud libraries not installed. Install with: pip install google-cloud-speech google-cloud-texttospeech")
            self.use_google_cloud = False
        except Exception as e:
            logger.error(f"❌ Failed to initialize Google Cloud clients: {e}")
            self.use_google_cloud = False
    
    async def speech_to_text(
        self,
        audio_data: bytes,
        sample_rate: int = 16000,
        language_code: str = "en-US"
    ) -> Optional[str]:
        """
        Convert speech audio to text
        
        Args:
            audio_data: Audio data bytes (WAV or WebM format)
            sample_rate: Audio sample rate in Hz (default: 16000)
            language_code: Language code (default: "en-US")
            
        Returns:
            Transcribed text or None if failed
        """
        if self.use_google_cloud and self.stt_client:
            return await self._google_stt(audio_data, sample_rate, language_code)
        else:
            # Web Speech API is handled client-side
            logger.info("ℹ️ Using Web Speech API (client-side STT)")
            return None
    
    async def _google_stt(
        self,
        audio_data: bytes,
        sample_rate: int,
        language_code: str
    ) -> Optional[str]:
        """
        Google Cloud Speech-to-Text
        
        Args:
            audio_data: Audio data bytes
            sample_rate: Sample rate in Hz
            language_code: Language code
            
        Returns:
            Transcribed text or None
        """
        try:
            from google.cloud import speech_v1 as speech
            
            audio = speech.RecognitionAudio(content=audio_data)
            config = speech.RecognitionConfig(
                encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
                sample_rate_hertz=sample_rate,
                language_code=language_code,
                enable_automatic_punctuation=True,
                model="latest_short"  # Optimized for short utterances
            )
            
            # Perform synchronous recognition
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                self.stt_client.recognize,
                config,
                audio
            )
            
            # Extract transcript
            if response.results:
                transcript = response.results[0].alternatives[0].transcript
                logger.info(f"🎤 STT: {transcript}")
                return transcript
            else:
                logger.warning("⚠️ No speech detected in audio")
                return None
                
        except Exception as e:
            logger.error(f"❌ Google STT error: {e}")
            return None
    
    async def text_to_speech(
        self,
        text: str,
        language_code: str = "en-US",
        voice_name: Optional[str] = None
    ) -> Optional[bytes]:
        """
        Convert text to speech audio
        
        Args:
            text: Text to convert
            language_code: Language code (default: "en-US")
            voice_name: Specific voice name (optional)
            
        Returns:
            Audio data bytes (MP3 format) or None if using Web Speech API
        """
        if self.use_google_cloud and self.tts_client:
            return await self._google_tts(text, language_code, voice_name)
        else:
            # Web Speech API is handled client-side
            logger.info("ℹ️ Using Web Speech API (client-side TTS)")
            return None
    
    async def _google_tts(
        self,
        text: str,
        language_code: str,
        voice_name: Optional[str]
    ) -> Optional[bytes]:
        """
        Google Cloud Text-to-Speech
        
        Args:
            text: Text to convert
            language_code: Language code
            voice_name: Voice name (optional)
            
        Returns:
            Audio data bytes (MP3)
        """
        try:
            from google.cloud import texttospeech
            
            # Set up synthesis input
            synthesis_input = texttospeech.SynthesisInput(text=text)
            
            # Voice configuration
            if voice_name:
                voice = texttospeech.VoiceSelectionParams(
                    name=voice_name,
                    language_code=language_code
                )
            else:
                voice = texttospeech.VoiceSelectionParams(
                    language_code=language_code,
                    ssml_gender=texttospeech.SsmlVoiceGender.NEUTRAL
                )
            
            # Audio configuration
            audio_config = texttospeech.AudioConfig(
                audio_encoding=texttospeech.AudioEncoding.MP3,
                speaking_rate=1.1  # Slightly faster for coaching
            )
            
            # Perform synthesis
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                self.tts_client.synthesize_speech,
                synthesis_input,
                voice,
                audio_config
            )
            
            logger.info(f"🔊 TTS: Generated {len(response.audio_content)} bytes")
            return response.audio_content
            
        except Exception as e:
            logger.error(f"❌ Google TTS error: {e}")
            return None
    
    def detect_voice_activity(
        self,
        audio_chunk: bytes,
        threshold: float = 0.02
    ) -> bool:
        """
        Simple energy-based voice activity detection
        
        Args:
            audio_chunk: Audio data chunk
            threshold: Energy threshold for voice detection
            
        Returns:
            True if voice activity detected
        """
        try:
            import numpy as np
            
            # Convert bytes to numpy array (assuming 16-bit PCM)
            audio_array = np.frombuffer(audio_chunk, dtype=np.int16)
            
            # Calculate RMS energy
            energy = np.sqrt(np.mean(audio_array.astype(np.float32) ** 2))
            
            # Normalize to 0-1 range
            normalized_energy = energy / 32768.0
            
            return normalized_energy > threshold
            
        except Exception as e:
            logger.error(f"❌ VAD error: {e}")
            return False
    
    def is_ready(self) -> bool:
        """
        Check if voice handler is ready
        
        Returns:
            True if ready for voice processing
        """
        if self.use_google_cloud:
            return self.stt_client is not None and self.tts_client is not None
        else:
            # Web Speech API is always ready (client-side)
            return True
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get voice handler statistics
        
        Returns:
            Dictionary with stats
        """
        return {
            'mode': 'Google Cloud' if self.use_google_cloud else 'Web Speech API',
            'stt_ready': self.stt_client is not None if self.use_google_cloud else True,
            'tts_ready': self.tts_client is not None if self.use_google_cloud else True,
            'ready': self.is_ready()
        }


class VoiceActivityDetector:
    """
    Voice Activity Detection using energy-based method
    Tracks speech start/end with silence detection
    """
    
    def __init__(
        self,
        speech_threshold: float = 0.02,
        silence_duration: float = 2.5
    ):
        """
        Initialize VAD
        
        Args:
            speech_threshold: Energy threshold for speech detection
            silence_duration: Seconds of silence to trigger speech end
        """
        self.speech_threshold = speech_threshold
        self.silence_duration = silence_duration
        
        self.is_speaking = False
        self.silence_start = None
        self.speech_start = None
    
    def process_audio_chunk(
        self,
        audio_chunk: bytes,
        timestamp: float
    ) -> Dict[str, Any]:
        """
        Process audio chunk and detect speech events
        
        Args:
            audio_chunk: Audio data chunk
            timestamp: Current timestamp
            
        Returns:
            Dictionary with VAD results and events
        """
        import numpy as np
        
        # Calculate energy
        try:
            audio_array = np.frombuffer(audio_chunk, dtype=np.int16)
            energy = np.sqrt(np.mean(audio_array.astype(np.float32) ** 2))
            normalized_energy = energy / 32768.0
        except:
            normalized_energy = 0.0
        
        # Detect speech
        has_speech = normalized_energy > self.speech_threshold
        
        result = {
            'energy': normalized_energy,
            'has_speech': has_speech,
            'is_speaking': self.is_speaking,
            'event': None
        }
        
        # State machine
        if has_speech:
            if not self.is_speaking:
                # Speech started
                self.is_speaking = True
                self.speech_start = timestamp
                result['event'] = 'speech_start'
                logger.debug("🎤 Speech started")
            
            # Reset silence timer
            self.silence_start = None
        
        else:  # No speech
            if self.is_speaking:
                # Start silence timer
                if self.silence_start is None:
                    self.silence_start = timestamp
                
                # Check if silence duration exceeded
                silence_elapsed = timestamp - self.silence_start
                if silence_elapsed >= self.silence_duration:
                    # Speech ended
                    self.is_speaking = False
                    result['event'] = 'speech_end'
                    result['speech_duration'] = timestamp - self.speech_start if self.speech_start else 0
                    logger.debug(f"🎤 Speech ended (duration: {result['speech_duration']:.1f}s)")
                    
                    # Reset timers
                    self.silence_start = None
                    self.speech_start = None
        
        return result
    
    def reset(self) -> None:
        """Reset VAD state"""
        self.is_speaking = False
        self.silence_start = None
        self.speech_start = None
