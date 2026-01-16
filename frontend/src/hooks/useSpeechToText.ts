/**
 * Speech-to-Text Hook
 * Uses Web Speech API for browser-native speech recognition
 */

import { useRef, useCallback } from 'react';

interface SpeechToTextConfig {
  language?: string;
  continuous?: boolean;
  interimResults?: boolean;
  onTranscript?: (transcript: string, isFinal: boolean) => void;
  onError?: (error: string) => void;
}

export const useSpeechToText = (config: SpeechToTextConfig = {}) => {
  const {
    language = 'en-US',
    continuous = false,
    interimResults = true,
    onTranscript,
    onError
  } = config;

  const recognitionRef = useRef<any>(null);
  const isListeningRef = useRef(false);

  const startRecognition = useCallback(() => {
    // Check if Web Speech API is available
    const SpeechRecognition = (window as any).SpeechRecognition || (window as any).webkitSpeechRecognition;

    if (!SpeechRecognition) {
      const error = 'Speech recognition not supported in this browser';
      console.error(error);
      onError?.(error);
      return false;
    }

    try {
      // Create recognition instance
      const recognition = new SpeechRecognition();
      recognition.lang = language;
      recognition.continuous = continuous;
      recognition.interimResults = interimResults;

      recognition.onstart = () => {
        isListeningRef.current = true;
        console.log('🎤 Speech recognition started');
      };

      recognition.onresult = (event: any) => {
        const results = event.results;
        const lastResult = results[results.length - 1];
        const transcript = lastResult[0].transcript;
        const isFinal = lastResult.isFinal;

        console.log(`🎤 Transcript (${isFinal ? 'final' : 'interim'}): ${transcript}`);
        onTranscript?.(transcript, isFinal);
      };

      recognition.onerror = (event: any) => {
        console.error('Speech recognition error:', event.error);
        onError?.(event.error);
        isListeningRef.current = false;
      };

      recognition.onend = () => {
        isListeningRef.current = false;
        console.log('🎤 Speech recognition ended');
      };

      recognition.start();
      recognitionRef.current = recognition;
      return true;
    } catch (error) {
      console.error('Failed to start speech recognition:', error);
      onError?.(String(error));
      return false;
    }
  }, [language, continuous, interimResults, onTranscript, onError]);

  const stopRecognition = useCallback(() => {
    if (recognitionRef.current && isListeningRef.current) {
      recognitionRef.current.stop();
      recognitionRef.current = null;
      isListeningRef.current = false;
    }
  }, []);

  const isListening = () => isListeningRef.current;

  return {
    startRecognition,
    stopRecognition,
    isListening
  };
};
