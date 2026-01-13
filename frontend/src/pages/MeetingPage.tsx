import React, { useState, useEffect, useRef, useCallback } from 'react';
import { Camera, Heart, Zap, Volume2, Play, Pause, Wifi, WifiOff, Copy, Check, Video, Home, AlertCircle } from 'lucide-react';
import { BACKEND_URL, WS_BASE_URL, POSE_PAIRS } from '../config';
import type { KeyPoint, AnalysisData, Stats, FeedbackItem, ConnectionStatus } from '../types';

interface MeetingPageProps {
  sessionId: string;
  onNavigate: (path: string) => void;
}

const MeetingPage: React.FC<MeetingPageProps> = ({ sessionId, onNavigate }) => {
  const [isStreaming, setIsStreaming] = useState(false);
  const [connectionStatus, setConnectionStatus] = useState<ConnectionStatus>('disconnected');
  const [stats, setStats] = useState<Stats>({
    frameCount: 0,
    balance: 0,
    energy: 'Unknown',
    emotion: 'Unknown',
    emotionConfidence: 0,
    posture: 'Unknown',
    postureAngle: 0
  });
  const [recentFeedback, setRecentFeedback] = useState<FeedbackItem[]>([]);
  const [lastAnalysis, setLastAnalysis] = useState<AnalysisData | null>(null);
  const [copied, setCopied] = useState(false);
  const [meetingLink, setMeetingLink] = useState('');
  const [error, setError] = useState<string | null>(null);

  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const overlayCanvasRef = useRef<HTMLCanvasElement>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const frameIntervalRef = useRef<number | null>(null);
  const reconnectTimeoutRef = useRef<number | null>(null);

  // Verify meeting on mount
  useEffect(() => {
    const verifyMeeting = async () => {
      try {
        const response = await fetch(`${BACKEND_URL}/api/meeting/${sessionId}`);
        if (!response.ok) {
          setError('Meeting not found or expired');
          setTimeout(() => onNavigate('/'), 2000);
          return;
        }
        const data = await response.json();
        setMeetingLink(`${window.location.origin}/meet/${sessionId}`);
        console.log('✅ Meeting verified:', data);
      } catch (err) {
        console.error('Error verifying meeting:', err);
        setError('Failed to verify meeting');
        setTimeout(() => onNavigate('/'), 2000);
      }
    };

    verifyMeeting();
  }, [sessionId, onNavigate]);

  // Draw pose skeleton on overlay canvas
  const drawSkeleton = useCallback((ctx: CanvasRenderingContext2D, keypoints: (KeyPoint | null)[]) => {
    if (!keypoints || keypoints.length === 0) return;

    const canvas = overlayCanvasRef.current;
    if (!canvas) return;

    // Clear canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Get video element for scaling
    const video = videoRef.current;
    if (!video) return;

    // Canvas is 640x480, keypoints are in this coordinate space
    const scaleX = canvas.width / 640;
    const scaleY = canvas.height / 480;

    // Draw connections first (lines between joints)
    ctx.strokeStyle = '#00ff00';
    ctx.lineWidth = 3;
    ctx.shadowBlur = 10;
    ctx.shadowColor = '#00ff00';

    POSE_PAIRS.forEach(([i, j]) => {
      const ptA = keypoints[i];
      const ptB = keypoints[j];
      if (ptA && ptB && ptA.confidence > 0.2 && ptB.confidence > 0.2) {
        ctx.beginPath();
        ctx.moveTo(ptA.x * scaleX, ptA.y * scaleY);
        ctx.lineTo(ptB.x * scaleX, ptB.y * scaleY);
        ctx.stroke();
      }
    });

    // Draw keypoints as circles
    ctx.shadowBlur = 15;
    keypoints.forEach((point) => {
      if (point && point.confidence > 0.2) {
        // Color code by confidence level
        const confidence = point.confidence;
        if (confidence > 0.7) {
          ctx.fillStyle = '#ff0000'; // High confidence - red
          ctx.shadowColor = '#ff0000';
        } else if (confidence > 0.5) {
          ctx.fillStyle = '#ffff00'; // Medium confidence - yellow
          ctx.shadowColor = '#ffff00';
        } else {
          ctx.fillStyle = '#ff8800'; // Low confidence - orange
          ctx.shadowColor = '#ff8800';
        }

        ctx.beginPath();
        ctx.arc(point.x * scaleX, point.y * scaleY, 6, 0, 2 * Math.PI);
        ctx.fill();
      }
    });

    ctx.shadowBlur = 0;
  }, []);

  // WebSocket connection with reconnection logic
  const connectWebSocket = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      console.log('WebSocket already connected');
      return;
    }

    setConnectionStatus('connecting');
    const wsUrl = `${WS_BASE_URL}/ws/meet/${sessionId}`;
    console.log(`🔌 Connecting to: ${wsUrl}`);

    const ws = new WebSocket(wsUrl);
    wsRef.current = ws;

    ws.onopen = () => {
      setConnectionStatus('connected');
      setError(null);
      console.log('✅ WebSocket connected to OpenPose backend');
    };

    ws.onmessage = (event: MessageEvent) => {
      try {
        const response = JSON.parse(event.data);

        // Handle welcome message
        if (response.type === 'welcome') {
          console.log('👋 Welcome:', response.message);
          return;
        }

        // Handle pong
        if (response.type === 'pong') {
          return;
        }

        // Handle analysis data from OpenPose
        if (response.type === 'analysis' && response.data) {
          const data: AnalysisData = response.data;
          setLastAnalysis(data);

          // Update stats from OpenPose analysis
          setStats({
            frameCount: data.frame_num || stats.frameCount,
            balance: data.balance?.balance_score || 0,
            energy: data.movement?.energy || 'Unknown',
            emotion: data.emotion?.emotion || 'Unknown',
            emotionConfidence: data.emotion?.confidence || 0,
            posture: data.posture?.status || 'Unknown',
            postureAngle: data.posture?.angle || 0
          });

          // Draw skeleton overlay from OpenPose keypoints
          const overlayCanvas = overlayCanvasRef.current;
          if (overlayCanvas && data.keypoints) {
            const ctx = overlayCanvas.getContext('2d');
            if (ctx) {
              drawSkeleton(ctx, data.keypoints);
            }
          }

          // Handle AI coaching feedback
          if (response.coaching?.triggered && response.coaching.feedback) {
            const timestamp = new Date().toLocaleTimeString();
            const newFeedback: FeedbackItem = {
              time: timestamp,
              text: response.coaching.feedback,
              reason: response.coaching.reason
            };

            setRecentFeedback(prev => [newFeedback, ...prev].slice(0, 5));

            // Speak feedback (optional)
            if ('speechSynthesis' in window) {
              const utterance = new SpeechSynthesisUtterance(response.coaching.feedback);
              utterance.rate = 1.0;
              utterance.pitch = 1.0;
              utterance.volume = 0.8;
              window.speechSynthesis.speak(utterance);
            }

            console.log('🎯 Coach:', response.coaching.feedback);
          }
        }

        // Handle error messages
        if (response.type === 'error') {
          console.error('Backend error:', response.message);
          setError(response.message);
        }

      } catch (err) {
        console.error('Failed to parse WebSocket message:', err);
      }
    };

    ws.onerror = (event) => {
      console.error('WebSocket error:', event);
      setConnectionStatus('error');
      setError('Connection error - check if backend is running on port 8000');
    };

    ws.onclose = (event) => {
      console.log('WebSocket closed:', event.code, event.reason);
      setConnectionStatus('disconnected');

      // Attempt reconnection if streaming was active
      if (isStreaming && !event.wasClean) {
        console.log('🔄 Attempting to reconnect in 3s...');
        reconnectTimeoutRef.current = window.setTimeout(() => {
          if (isStreaming) {
            connectWebSocket();
          }
        }, 3000);
      }
    };
  }, [sessionId, isStreaming, stats.frameCount, drawSkeleton]);

  // Capture and send video frames to OpenPose backend
  const captureAndSendFrame = useCallback(() => {
    const canvas = canvasRef.current;
    const video = videoRef.current;
    const ws = wsRef.current;

    if (!canvas || !video || !ws || ws.readyState !== WebSocket.OPEN) {
      return;
    }

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Draw current video frame to canvas (640x480)
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    // Convert to base64 JPEG for transmission
    const frameData = canvas.toDataURL('image/jpeg', 0.8);

    try {
      ws.send(JSON.stringify({
        type: 'frame',
        frame: frameData,
        timestamp: Date.now() / 1000
      }));
    } catch (err) {
      console.error('Error sending frame:', err);
    }
  }, []);

  // Start video streaming
  const startStreaming = async () => {
    try {
      setError(null);

      // Get camera access
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          width: { ideal: 640 },
          height: { ideal: 480 },
          facingMode: 'user'
        },
        audio: false
      });

      streamRef.current = stream;

      // Attach to video element
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
      }

      // Setup canvas sizes (640x480 for OpenPose)
      if (canvasRef.current && overlayCanvasRef.current) {
        canvasRef.current.width = 640;
        canvasRef.current.height = 480;
        overlayCanvasRef.current.width = 640;
        overlayCanvasRef.current.height = 480;
      }

      // Connect WebSocket to OpenPose backend
      connectWebSocket();

      // Start sending frames at 10 FPS (every 100ms)
      frameIntervalRef.current = window.setInterval(captureAndSendFrame, 100);

      setIsStreaming(true);
      console.log('🎥 Streaming started - sending frames to OpenPose backend');

    } catch (err) {
      console.error('Camera access error:', err);
      setError('Could not access camera. Please allow camera permissions.');
    }
  };

  // Stop video streaming
  const stopStreaming = () => {
    console.log('🛑 Stopping stream...');

    // Clear frame capture interval
    if (frameIntervalRef.current) {
      clearInterval(frameIntervalRef.current);
      frameIntervalRef.current = null;
    }

    // Clear reconnection timeout
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current);
      reconnectTimeoutRef.current = null;
    }

    // Stop media stream
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => track.stop());
      streamRef.current = null;
    }

    // Close WebSocket
    if (wsRef.current) {
      if (wsRef.current.readyState === WebSocket.OPEN) {
        wsRef.current.send(JSON.stringify({ type: 'end' }));
      }
      wsRef.current.close();
      wsRef.current = null;
    }

    // Clear video element
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }

    // Stop speech synthesis
    if ('speechSynthesis' in window) {
      window.speechSynthesis.cancel();
    }

    setIsStreaming(false);
    setConnectionStatus('disconnected');
    console.log('✅ Stream stopped');
  };

  // Copy meeting link to clipboard
  const copyMeetingLink = () => {
    navigator.clipboard.writeText(meetingLink);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  // Get color for posture status
  const getPostureColor = (posture: string) => {
    switch (posture) {
      case 'Excellent': return 'text-green-400';
      case 'Good': return 'text-blue-400';
      case 'Fair': return 'text-yellow-400';
      default: return 'text-red-400';
    }
  };

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      stopStreaming();
    };
  }, []);

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900 text-white">
      {/* Header */}
      <header className="bg-black/30 backdrop-blur-sm border-b border-purple-500/20">
        <div className="max-w-7xl mx-auto px-4 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-4">
              <button
                onClick={() => { stopStreaming(); onNavigate('/'); }}
                className="p-2 hover:bg-white/10 rounded-lg transition"
              >
                <Home className="w-5 h-5" />
              </button>
              <div className="flex items-center gap-3">
                <div className="bg-purple-600 p-2 rounded-lg">
                  <Video className="w-6 h-6" />
                </div>
                <div>
                  <h1 className="text-lg font-bold">AI Video Coach Meeting</h1>
                  <p className="text-xs text-purple-300 font-mono">
                    {sessionId.slice(0, 12)}...
                  </p>
                </div>
              </div>
            </div>

            <div className="flex items-center gap-3">
              <button
                onClick={copyMeetingLink}
                className="flex items-center gap-2 px-3 py-2 bg-purple-600/20 hover:bg-purple-600/30 border border-purple-500/30 rounded-lg transition text-sm"
              >
                {copied ? (
                  <>
                    <Check className="w-4 h-4" />
                    Copied!
                  </>
                ) : (
                  <>
                    <Copy className="w-4 h-4" />
                    Share Link
                  </>
                )}
              </button>

              <div className={`flex items-center gap-2 px-3 py-1 rounded-full text-sm ${connectionStatus === 'connected' ? 'bg-green-500/20 text-green-400' :
                  connectionStatus === 'connecting' ? 'bg-yellow-500/20 text-yellow-400' :
                    connectionStatus === 'error' ? 'bg-red-500/20 text-red-400' :
                      'bg-gray-500/20 text-gray-400'
                }`}>
                {connectionStatus === 'connected' ? (
                  <Wifi className="w-4 h-4" />
                ) : (
                  <WifiOff className="w-4 h-4" />
                )}
                {connectionStatus}
              </div>
            </div>
          </div>
        </div>
      </header>

      {/* Error Banner */}
      {error && (
        <div className="bg-red-500/20 border-b border-red-500/30 px-4 py-3">
          <div className="max-w-7xl mx-auto flex items-center gap-2 text-red-200">
            <AlertCircle className="w-5 h-5" />
            <span>{error}</span>
          </div>
        </div>
      )}

      {/* Main Content */}
      <div className="max-w-7xl mx-auto p-6">
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">

          {/* Video Feed Section */}
          <div className="lg:col-span-2 space-y-4">
            <div className="bg-black/40 backdrop-blur-sm rounded-2xl border border-purple-500/20 overflow-hidden">
              <div className="relative aspect-video bg-black">
                {/* Video Element */}
                <video
                  ref={videoRef}
                  autoPlay
                  playsInline
                  muted
                  className="absolute inset-0 w-full h-full object-cover"
                />

                {/* Hidden canvas for frame capture */}
                <canvas ref={canvasRef} className="hidden" />

                {/* Overlay canvas for OpenPose skeleton */}
                <canvas
                  ref={overlayCanvasRef}
                  className="absolute inset-0 w-full h-full pointer-events-none"
                />

                {/* Camera placeholder when not streaming */}
                {!isStreaming && (
                  <div className="absolute inset-0 flex items-center justify-center bg-black/60 backdrop-blur-sm">
                    <div className="text-center">
                      <Camera className="w-16 h-16 mx-auto mb-4 text-purple-400" />
                      <p className="text-lg mb-2">Ready to Start</p>
                      <p className="text-sm text-gray-400">OpenPose AI Coach Ready</p>
                    </div>
                  </div>
                )}

                {/* Live stats overlay */}
                {isStreaming && (
                  <>
                    <div className="absolute top-4 left-4 space-y-2">
                      <div className="bg-black/70 backdrop-blur-sm px-3 py-2 rounded-lg">
                        <div className="text-xs text-purple-300">Balance</div>
                        <div className="text-lg font-bold">{stats.balance.toFixed(0)}/100</div>
                      </div>
                      <div className="bg-black/70 backdrop-blur-sm px-3 py-2 rounded-lg">
                        <div className="text-xs text-purple-300">Emotion</div>
                        <div className="text-sm font-bold">{stats.emotion}</div>
                        <div className="text-xs text-gray-400">{stats.emotionConfidence.toFixed(0)}%</div>
                      </div>
                      <div className="bg-black/70 backdrop-blur-sm px-3 py-2 rounded-lg">
                        <div className="text-xs text-purple-300">Posture</div>
                        <div className={`text-sm font-bold ${getPostureColor(stats.posture)}`}>
                          {stats.posture}
                        </div>
                        <div className="text-xs text-gray-400">{stats.postureAngle.toFixed(1)}°</div>
                      </div>
                    </div>

                    <div className="absolute top-4 right-4 bg-black/70 backdrop-blur-sm px-3 py-2 rounded-lg">
                      <div className="text-xs text-purple-300">Frame</div>
                      <div className="text-lg font-bold">{stats.frameCount}</div>
                    </div>
                  </>
                )}
              </div>

              {/* Controls */}
              <div className="p-4 bg-black/20 border-t border-purple-500/20">
                <div className="flex items-center justify-center gap-4">
                  {!isStreaming ? (
                    <button
                      onClick={startStreaming}
                      className="flex items-center gap-2 px-6 py-3 bg-purple-600 hover:bg-purple-700 rounded-xl font-semibold transition"
                    >
                      <Play className="w-5 h-5" />
                      Start Session
                    </button>
                  ) : (
                    <button
                      onClick={stopStreaming}
                      className="flex items-center gap-2 px-6 py-3 bg-red-600 hover:bg-red-700 rounded-xl font-semibold transition"
                    >
                      <Pause className="w-5 h-5" />
                      Stop Session
                    </button>
                  )}
                </div>
              </div>
            </div>

            {/* Detailed Analysis */}
            {lastAnalysis && (
              <div className="bg-black/40 backdrop-blur-sm rounded-2xl border border-purple-500/20 p-6">
                <h3 className="text-lg font-bold mb-4">OpenPose Analysis</h3>
                <div className="grid grid-cols-2 md:grid-cols-3 gap-4 text-sm">
                  <div>
                    <span className="text-gray-400">Posture Angle:</span>
                    <span className="ml-2 font-semibold">{stats.postureAngle.toFixed(1)}°</span>
                  </div>
                  <div>
                    <span className="text-gray-400">Movement Score:</span>
                    <span className="ml-2 font-semibold">
                      {lastAnalysis.movement.movement_score.toFixed(2)}
                    </span>
                  </div>
                  <div>
                    <span className="text-gray-400">Velocity:</span>
                    <span className="ml-2 font-semibold">
                      {lastAnalysis.movement.velocity.toFixed(2)} px/f
                    </span>
                  </div>
                  <div>
                    <span className="text-gray-400">Arm Symmetry:</span>
                    <span className="ml-2 font-semibold">
                      {lastAnalysis.symmetry.arm_symmetry?.toFixed(1) || 'N/A'}%
                    </span>
                  </div>
                  <div>
                    <span className="text-gray-400">Leg Symmetry:</span>
                    <span className="ml-2 font-semibold">
                      {lastAnalysis.symmetry.leg_symmetry?.toFixed(1) || 'N/A'}%
                    </span>
                  </div>
                  <div>
                    <span className="text-gray-400">Activities:</span>
                    <span className="ml-2 font-semibold">
                      {lastAnalysis.activities.join(', ')}
                    </span>
                  </div>
                </div>
              </div>
            )}
          </div>

          {/* Stats & Feedback Panel */}
          <div className="space-y-4">
            {/* Live Metrics */}
            <div className="bg-black/40 backdrop-blur-sm rounded-2xl border border-purple-500/20 p-6">
              <h2 className="text-lg font-bold mb-4 flex items-center gap-2">
                <Zap className="w-5 h-5 text-yellow-400" />
                Live Metrics
              </h2>

              <div className="space-y-4">
                <div>
                  <div className="flex justify-between text-sm mb-1">
                    <span className="text-gray-400">Balance</span>
                    <span className="font-semibold">{stats.balance.toFixed(0)}/100</span>
                  </div>
                  <div className="h-2 bg-gray-700 rounded-full overflow-hidden">
                    <div
                      className="h-full bg-gradient-to-r from-purple-500 to-pink-500 transition-all duration-300"
                      style={{ width: `${stats.balance}%` }}
                    />
                  </div>
                </div>

                <div>
                  <div className="flex justify-between text-sm mb-1">
                    <span className="text-gray-400">Energy</span>
                    <span className="font-semibold">{stats.energy}</span>
                  </div>
                </div>

                <div>
                  <div className="flex justify-between text-sm mb-1">
                    <span className="text-gray-400">Posture</span>
                    <span className={`font-semibold ${getPostureColor(stats.posture)}`}>
                      {stats.posture}
                    </span>
                  </div>
                </div>

                <div>
                  <div className="flex justify-between text-sm mb-1">
                    <span className="text-gray-400">Emotion</span>
                    <span className="font-semibold">{stats.emotion}</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <Heart className="w-4 h-4 text-pink-400" />
                    <span className="text-sm">{stats.emotionConfidence.toFixed(0)}%</span>
                  </div>
                </div>
              </div>
            </div>

            {/* AI Coach Feedback */}
            <div className="bg-black/40 backdrop-blur-sm rounded-2xl border border-purple-500/20 p-6">
              <h2 className="text-lg font-bold mb-4 flex items-center gap-2">
                <Volume2 className="w-5 h-5 text-purple-400" />
                AI Coach Feedback
              </h2>

              {recentFeedback.length === 0 ? (
                <div className="text-center py-8 text-gray-400">
                  <Volume2 className="w-8 h-8 mx-auto mb-2 opacity-50" />
                  <p className="text-sm">No feedback yet</p>
                  <p className="text-xs mt-1">Start moving to receive coaching</p>
                </div>
              ) : (
                <div className="space-y-3 max-h-[400px] overflow-y-auto">
                  {recentFeedback.map((feedback, i) => (
                    <div
                      key={`${feedback.time}-${i}`}
                      className="bg-purple-500/10 border border-purple-500/20 rounded-lg p-3 animate-fade-in"
                    >
                      <div className="flex items-start justify-between gap-2 mb-2">
                        <div className="text-xs text-purple-300">{feedback.time}</div>
                        {feedback.reason && (
                          <div className="text-xs px-2 py-0.5 bg-purple-500/20 rounded">
                            {feedback.reason.replace(/_/g, ' ')}
                          </div>
                        )}
                      </div>
                      <div className="text-sm leading-relaxed">{feedback.text}</div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default MeetingPage;