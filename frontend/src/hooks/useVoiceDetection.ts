/**
 * Voice Activity Detection Hook
 * Detects when user is speaking using Web Audio API
 */

import { useEffect, useRef, useState, useCallback } from 'react';

interface VoiceDetectionConfig {
  speechThreshold?: number;
  silenceDuration?: number; // milliseconds
  onSpeechStart?: () => void;
  onSpeechEnd?: (duration: number) => void;
}

interface VoiceDetectionState {
  isSpeaking: boolean;
  energy: number;
  isListening: boolean;
}

export const useVoiceDetection = (config: VoiceDetectionConfig = {}) => {
  const {
    speechThreshold = 0.02,
    silenceDuration = 2500, // 2.5 seconds
    onSpeechStart,
    onSpeechEnd
  } = config;

  const [state, setState] = useState<VoiceDetectionState>({
    isSpeaking: false,
    energy: 0,
    isListening: false
  });

  const audioContextRef = useRef<AudioContext | null>(null);
  const analyserRef = useRef<AnalyserNode | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const animationFrameRef = useRef<number | null>(null);
  const silenceTimerRef = useRef<number | null>(null);
  const speechStartTimeRef = useRef<number | null>(null);

  const analyzeAudio = useCallback(() => {
    if (!analyserRef.current) return;

    const analyser = analyserRef.current;
    const dataArray = new Uint8Array(analyser.frequencyBinCount);
    analyser.getByteFrequencyData(dataArray);

    // Calculate average volume
    const average = dataArray.reduce((sum, value) => sum + value, 0) / dataArray.length;
    const normalizedEnergy = average / 255;

    setState(prev => ({ ...prev, energy: normalizedEnergy }));

    // Detect speech
    const hasSpeech = normalizedEnergy > speechThreshold;

    if (hasSpeech) {
      setState(prev => {
        if (!prev.isSpeaking) {
          // Speech started
          speechStartTimeRef.current = Date.now();
          onSpeechStart?.();
          return { ...prev, isSpeaking: true };
        }
        return prev;
      });

      // Clear silence timer
      if (silenceTimerRef.current) {
        clearTimeout(silenceTimerRef.current);
        silenceTimerRef.current = null;
      }
    } else {
      // No speech detected
      setState(prev => {
        if (prev.isSpeaking && !silenceTimerRef.current) {
          // Start silence timer
          silenceTimerRef.current = setTimeout(() => {
            const duration = speechStartTimeRef.current
              ? Date.now() - speechStartTimeRef.current
              : 0;

            setState(s => ({ ...s, isSpeaking: false }));
            onSpeechEnd?.(duration);
            silenceTimerRef.current = null;
            speechStartTimeRef.current = null;
          }, silenceDuration);
        }
        return prev;
      });
    }

    // Continue analyzing
    animationFrameRef.current = requestAnimationFrame(analyzeAudio);
  }, [speechThreshold, silenceDuration, onSpeechStart, onSpeechEnd]);

  const startListening = useCallback(async () => {
    try {
      // Get microphone access
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true
        }
      });
      streamRef.current = stream;

      // Create audio context
      const audioContext = new AudioContext();
      audioContextRef.current = audioContext;

      // Create analyser
      const analyser = audioContext.createAnalyser();
      analyser.fftSize = 2048;
      analyser.smoothingTimeConstant = 0.8;
      analyserRef.current = analyser;

      // Connect stream to analyser
      const source = audioContext.createMediaStreamSource(stream);
      source.connect(analyser);

      setState(prev => ({ ...prev, isListening: true }));

      // Start analyzing
      analyzeAudio();

      console.log('🎤 Voice detection started');
    } catch (error) {
      console.error('Failed to start voice detection:', error);
      throw error;
    }
  }, [analyzeAudio]);

  const stopListening = useCallback(() => {
    // Stop animation frame
    if (animationFrameRef.current) {
      cancelAnimationFrame(animationFrameRef.current);
      animationFrameRef.current = null;
    }

    // Clear silence timer
    if (silenceTimerRef.current) {
      clearTimeout(silenceTimerRef.current);
      silenceTimerRef.current = null;
    }

    // Stop audio stream
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => track.stop());
      streamRef.current = null;
    }

    // Close audio context
    if (audioContextRef.current) {
      audioContextRef.current.close();
      audioContextRef.current = null;
    }

    analyserRef.current = null;
    setState({ isSpeaking: false, energy: 0, isListening: false });

    console.log('🎤 Voice detection stopped');
  }, []);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      stopListening();
    };
  }, [stopListening]);

  return {
    ...state,
    startListening,
    stopListening
  };
};
