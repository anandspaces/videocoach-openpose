import React, { useState, useEffect, useRef, useCallback } from 'react';
import { Camera, Activity, Heart, Zap, Volume2, VolumeX, Play, Pause, Settings, AlertCircle, CheckCircle, Wifi, WifiOff } from 'lucide-react';

// TypeScript Interfaces
interface KeyPoint {
  x: number;
  y: number;
  confidence: number;
}

interface AnalysisData {
  frame_num: number;
  timestamp: number;
  keypoints: (KeyPoint | null)[];
  joints: {
    right_elbow?: number;
    left_elbow?: number;
    right_knee?: number;
    left_knee?: number;
    right_hip?: number;
    left_hip?: number;
  };
  symmetry: {
    shoulder_width?: number;
    arm_symmetry?: number;
    leg_symmetry?: number;
    hip_width?: number;
  };
  balance: {
    cog: [number, number];
    balance_score: number;
  };
  movement: {
    energy: string;
    movement_score: number;
    velocity: number;
    sentiment?: string;
  };
  emotion: {
    emotion: string;
    confidence: number;
    sentiment: string;
    details?: string;
  };
  posture: {
    status: string;
    angle: number;
    shoulder_aligned?: boolean;
  };
  activities: string[];
}

interface Stats {
  frameCount: number;
  balance: number;
  energy: string;
  emotion: string;
  emotionConfidence: number;
  posture: string;
  postureAngle: number;
}

interface FeedbackItem {
  time: string;
  text: string;
  reason?: string;
}

type ConnectionStatus = 'disconnected' | 'connected' | 'error' | 'connecting';

const POSE_PAIRS = [[0, 1], [1, 2], [2, 3], [3, 4], [1, 5], [5, 6], [6, 7], [1, 8], [8, 9], [9, 10], [1, 11], [11, 12], [12, 13], [0, 14], [0, 15], [14, 16], [15, 17]];

const VideoCoachApp: React.FC = () => {
  const [isStreaming, setIsStreaming] = useState<boolean>(false);
  const [isAudioEnabled, setIsAudioEnabled] = useState<boolean>(true);
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
  const [showSettings, setShowSettings] = useState<boolean>(false);
  const [backendUrl, setBackendUrl] = useState<string>('ws://localhost:8000');
  const [frameRate, setFrameRate] = useState<number>(10);
  const [lastAnalysis, setLastAnalysis] = useState<AnalysisData | null>(null);

  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const overlayCanvasRef = useRef<HTMLCanvasElement>(null);
  const wsVideoRef = useRef<WebSocket | null>(null);
  const wsAudioRef = useRef<WebSocket | null>(null);
  const audioContextRef = useRef<AudioContext | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const frameIntervalRef = useRef<number | null>(null);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      stopStreaming();
    };
  }, []);

  const drawSkeleton = useCallback((ctx: CanvasRenderingContext2D, keypoints: (KeyPoint | null)[]) => {
    if (!keypoints || keypoints.length === 0) return;

    // Draw skeleton lines
    ctx.strokeStyle = '#00ff00';
    ctx.lineWidth = 3;

    POSE_PAIRS.forEach(([i, j]) => {
      const ptA = keypoints[i];
      const ptB = keypoints[j];

      if (ptA && ptB) {
        ctx.beginPath();
        ctx.moveTo(ptA.x, ptA.y);
        ctx.lineTo(ptB.x, ptB.y);
        ctx.stroke();
      }
    });

    // Draw keypoints
    keypoints.forEach((point) => {
      if (point && point.confidence > 0.3) {
        ctx.fillStyle = '#ff0000';
        ctx.beginPath();
        ctx.arc(point.x, point.y, 5, 0, 2 * Math.PI);
        ctx.fill();
      }
    });
  }, []);

  const connectWebSockets = useCallback(() => {
    setConnectionStatus('connecting');

    // Video analysis WebSocket
    wsVideoRef.current = new WebSocket(`${backendUrl}/ws/video-analysis`);

    wsVideoRef.current.onopen = () => {
      setConnectionStatus('connected');
      console.log('✅ Connected to backend');
    };

    wsVideoRef.current.onmessage = (event: MessageEvent) => {
      try {
        const response = JSON.parse(event.data);

        // Handle acknowledgment
        if (response.status === 'received') {
          return;
        }

        // Handle analysis response (if backend processes frames)
        if (response.keypoints) {
          setLastAnalysis(response);

          // Update stats
          setStats({
            frameCount: response.frame_num || stats.frameCount,
            balance: response.balance?.balance_score || 0,
            energy: response.movement?.energy || 'Unknown',
            emotion: response.emotion?.emotion || 'Unknown',
            emotionConfidence: response.emotion?.confidence || 0,
            posture: response.posture?.status || 'Unknown',
            postureAngle: response.posture?.angle || 0
          });

          // Draw skeleton on overlay
          const overlayCanvas = overlayCanvasRef.current;
          if (overlayCanvas && response.keypoints) {
            const ctx = overlayCanvas.getContext('2d');
            if (ctx) {
              ctx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);
              drawSkeleton(ctx, response.keypoints);
            }
          }
        }
      } catch (err) {
        console.error('Failed to parse backend response:', err);
      }
    };

    wsVideoRef.current.onerror = (error: Event) => {
      console.error('WebSocket error:', error);
      setConnectionStatus('error');
    };

    wsVideoRef.current.onclose = () => {
      setConnectionStatus('disconnected');
    };

    // Audio WebSocket
    if (isAudioEnabled) {
      wsAudioRef.current = new WebSocket(`${backendUrl}/ws/coach-audio`);

      const AudioContextClass = window.AudioContext || (window as any).webkitAudioContext;
      audioContextRef.current = new AudioContextClass();

      wsAudioRef.current.onmessage = async (event: MessageEvent) => {
        if (event.data instanceof Blob && audioContextRef.current) {
          try {
            const arrayBuffer = await event.data.arrayBuffer();
            const audioBuffer = await audioContextRef.current.decodeAudioData(arrayBuffer);

            const source = audioContextRef.current.createBufferSource();
            source.buffer = audioBuffer;
            source.connect(audioContextRef.current.destination);
            source.start(0);

            const timestamp = new Date().toLocaleTimeString();
            setRecentFeedback(prev => [{
              time: timestamp,
              text: 'Coach feedback received'
            }, ...prev].slice(0, 5));
          } catch (err) {
            console.error('Audio playback error:', err);
          }
        }
      };

      wsAudioRef.current.onerror = (error: Event) => {
        console.error('Audio WebSocket error:', error);
      };
    }
  }, [backendUrl, isAudioEnabled, stats.frameCount, drawSkeleton]);

  const captureAndSendFrame = useCallback(() => {
    const canvas = canvasRef.current;
    const video = videoRef.current;

    if (!canvas || !video || !wsVideoRef.current || wsVideoRef.current.readyState !== WebSocket.OPEN) {
      return;
    }

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Draw current video frame to canvas
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    // Convert to base64 JPEG
    const frameData = canvas.toDataURL('image/jpeg', 0.8);
    const base64Data = frameData.split(',')[1];

    // Send to backend for processing
    try {
      wsVideoRef.current.send(JSON.stringify({
        frame: base64Data,
        timestamp: Date.now() / 1000
      }));

      setStats(prev => ({
        ...prev,
        frameCount: prev.frameCount + 1
      }));
    } catch (err) {
      console.error('Error sending frame:', err);
    }
  }, []);

  const startStreaming = async (): Promise<void> => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          width: 640,
          height: 480,
          facingMode: 'user'
        }
      });

      streamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
      }

      connectWebSockets();
      setIsStreaming(true);

      // Start frame capture at specified frame rate
      const interval = 1000 / frameRate;
      frameIntervalRef.current = setInterval(captureAndSendFrame, interval);
    } catch (err) {
      console.error('Camera access error:', err);
      alert('Could not access camera. Please grant permission and ensure camera is not in use.');
    }
  };

  const stopStreaming = (): void => {
    if (frameIntervalRef.current) {
      clearInterval(frameIntervalRef.current);
      frameIntervalRef.current = null;
    }

    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => track.stop());
      streamRef.current = null;
    }

    if (wsVideoRef.current) {
      wsVideoRef.current.close();
      wsVideoRef.current = null;
    }

    if (wsAudioRef.current) {
      wsAudioRef.current.close();
      wsAudioRef.current = null;
    }

    if (audioContextRef.current) {
      audioContextRef.current.close();
      audioContextRef.current = null;
    }

    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }

    setIsStreaming(false);
    setConnectionStatus('disconnected');
    setLastAnalysis(null);

    // Clear overlay
    const overlayCanvas = overlayCanvasRef.current;
    if (overlayCanvas) {
      const ctx = overlayCanvas.getContext('2d');
      if (ctx) {
        ctx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);
      }
    }
  };

  const getPostureColor = (posture: string): string => {
    switch (posture) {
      case 'Excellent': return 'text-green-400';
      case 'Good': return 'text-blue-400';
      case 'Fair': return 'text-yellow-400';
      default: return 'text-red-400';
    }
  };

  const getPostureBgClass = (posture: string): string => {
    switch (posture) {
      case 'Excellent': return 'bg-green-500/20 border-green-500/30';
      case 'Good': return 'bg-blue-500/20 border-blue-500/30';
      case 'Fair': return 'bg-yellow-500/20 border-yellow-500/30';
      default: return 'bg-red-500/20 border-red-500/30';
    }
  };

  const getConnectionIcon = (status: ConnectionStatus) => {
    return status === 'connected' ? <Wifi className="w-4 h-4" /> : <WifiOff className="w-4 h-4" />;
  };

  const getConnectionColor = (status: ConnectionStatus): string => {
    switch (status) {
      case 'connected': return 'text-green-400';
      case 'error': return 'text-red-400';
      case 'connecting': return 'text-yellow-400';
      default: return 'text-gray-400';
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900 text-white">
      {/* Header */}
      <header className="bg-black/30 backdrop-blur-sm border-b border-purple-500/20">
        <div className="max-w-7xl mx-auto px-4 py-4 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="bg-purple-600 p-2 rounded-lg">
              <Activity className="w-6 h-6" />
            </div>
            <div>
              <h1 className="text-xl font-bold">AI Video Coach</h1>
              <p className="text-xs text-purple-300">Real-time Pose & Emotion Analysis</p>
            </div>
          </div>

          <div className="flex items-center gap-4">
            <div className={`flex items-center gap-2 px-3 py-1 rounded-full text-sm ${getConnectionColor(connectionStatus)}`}>
              {getConnectionIcon(connectionStatus)}
              {connectionStatus}
            </div>

            <button
              onClick={() => setShowSettings(!showSettings)}
              className="p-2 hover:bg-white/10 rounded-lg transition"
              aria-label="Settings"
            >
              <Settings className="w-5 h-5" />
            </button>
          </div>
        </div>
      </header>

      <div className="max-w-7xl mx-auto p-6">
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Main Video Feed */}
          <div className="lg:col-span-2 space-y-4">
            <div className="bg-black/40 backdrop-blur-sm rounded-2xl border border-purple-500/20 overflow-hidden">
              <div className="relative aspect-video bg-black">
                <video
                  ref={videoRef}
                  autoPlay
                  playsInline
                  muted
                  className="absolute inset-0 w-full h-full object-cover"
                />
                <canvas
                  ref={canvasRef}
                  width={640}
                  height={480}
                  className="hidden"
                />
                <canvas
                  ref={overlayCanvasRef}
                  width={640}
                  height={480}
                  className="absolute inset-0 w-full h-full pointer-events-none"
                />

                {!isStreaming && (
                  <div className="absolute inset-0 flex items-center justify-center bg-black/60 backdrop-blur-sm">
                    <div className="text-center">
                      <Camera className="w-16 h-16 mx-auto mb-4 text-purple-400" />
                      <p className="text-lg mb-2">Camera Ready</p>
                      <p className="text-sm text-gray-400">Click Start to begin analysis</p>
                    </div>
                  </div>
                )}

                {/* Live Stats Overlay */}
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
                      type="button"
                    >
                      <Play className="w-5 h-5" />
                      Start Session
                    </button>
                  ) : (
                    <button
                      onClick={stopStreaming}
                      className="flex items-center gap-2 px-6 py-3 bg-red-600 hover:bg-red-700 rounded-xl font-semibold transition"
                      type="button"
                    >
                      <Pause className="w-5 h-5" />
                      Stop Session
                    </button>
                  )}

                  <button
                    onClick={() => setIsAudioEnabled(!isAudioEnabled)}
                    className={`p-3 rounded-xl transition ${isAudioEnabled ? 'bg-green-600 hover:bg-green-700' : 'bg-gray-600 hover:bg-gray-700'
                      }`}
                    type="button"
                    aria-label={isAudioEnabled ? 'Disable audio' : 'Enable audio'}
                  >
                    {isAudioEnabled ? <Volume2 className="w-5 h-5" /> : <VolumeX className="w-5 h-5" />}
                  </button>
                </div>
              </div>
            </div>

            {/* Analysis Details */}
            {lastAnalysis && (
              <div className="bg-black/40 backdrop-blur-sm rounded-2xl border border-purple-500/20 p-6">
                <h3 className="text-lg font-bold mb-4">Current Analysis</h3>
                <div className="grid grid-cols-2 gap-4 text-sm">
                  <div>
                    <span className="text-gray-400">Posture Angle:</span>
                    <span className="ml-2 font-semibold">{stats.postureAngle.toFixed(1)}°</span>
                  </div>
                  <div>
                    <span className="text-gray-400">Movement Score:</span>
                    <span className="ml-2 font-semibold">{lastAnalysis.movement.movement_score.toFixed(2)}</span>
                  </div>
                  <div>
                    <span className="text-gray-400">Velocity:</span>
                    <span className="ml-2 font-semibold">{lastAnalysis.movement.velocity.toFixed(2)} px/f</span>
                  </div>
                  <div>
                    <span className="text-gray-400">Arm Symmetry:</span>
                    <span className="ml-2 font-semibold">{lastAnalysis.symmetry.arm_symmetry?.toFixed(1) || 'N/A'}%</span>
                  </div>
                </div>
              </div>
            )}
          </div>

          {/* Stats Panel */}
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
                    <span className="text-gray-400">Energy Level</span>
                    <span className="font-semibold">{stats.energy}</span>
                  </div>
                  <div className="px-3 py-2 bg-blue-500/20 border border-blue-500/30 rounded-lg text-sm">
                    {stats.energy}
                  </div>
                </div>

                <div>
                  <div className="flex justify-between text-sm mb-1">
                    <span className="text-gray-400">Posture</span>
                    <span className="font-semibold">{stats.posture}</span>
                  </div>
                  <div className={`px-3 py-2 border rounded-lg text-sm ${getPostureBgClass(stats.posture)}`}>
                    {stats.posture}
                  </div>
                </div>

                <div>
                  <div className="flex justify-between text-sm mb-1">
                    <span className="text-gray-400">Emotion</span>
                    <span className="font-semibold">{stats.emotionConfidence.toFixed(0)}%</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <Heart className="w-4 h-4 text-pink-400" />
                    <span className="text-sm">{stats.emotion}</span>
                  </div>
                </div>
              </div>
            </div>

            {/* Recent Feedback */}
            <div className="bg-black/40 backdrop-blur-sm rounded-2xl border border-purple-500/20 p-6">
              <h2 className="text-lg font-bold mb-4">Coach Feedback</h2>

              {recentFeedback.length === 0 ? (
                <div className="text-center py-8 text-gray-400">
                  <Volume2 className="w-8 h-8 mx-auto mb-2 opacity-50" />
                  <p className="text-sm">No feedback yet</p>
                  <p className="text-xs mt-1">Start exercising to receive coaching</p>
                </div>
              ) : (
                <div className="space-y-3">
                  {recentFeedback.map((feedback, i) => (
                    <div
                      key={`${feedback.time}-${i}`}
                      className="bg-purple-500/10 border border-purple-500/20 rounded-lg p-3"
                    >
                      <div className="text-xs text-purple-300 mb-1">{feedback.time}</div>
                      <div className="text-sm">{feedback.text}</div>
                      {feedback.reason && (
                        <div className="text-xs text-gray-400 mt-1">Reason: {feedback.reason}</div>
                      )}
                    </div>
                  ))}
                </div>
              )}
            </div>

            {/* System Status */}
            <div className="bg-black/40 backdrop-blur-sm rounded-2xl border border-purple-500/20 p-6">
              <h2 className="text-lg font-bold mb-4">System Status</h2>
              <div className="space-y-2 text-sm">
                <div className="flex justify-between items-center">
                  <span className="text-gray-400">Backend</span>
                  <span className={connectionStatus === 'connected' ? 'text-green-400 flex items-center gap-1' : 'text-red-400 flex items-center gap-1'}>
                    {connectionStatus === 'connected' ? <CheckCircle className="w-4 h-4" /> : <AlertCircle className="w-4 h-4" />}
                    {connectionStatus === 'connected' ? 'Connected' : 'Disconnected'}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-400">Audio</span>
                  <span className={isAudioEnabled ? 'text-green-400' : 'text-gray-400'}>
                    {isAudioEnabled ? '✓ Enabled' : '✗ Disabled'}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-400">Frames Processed</span>
                  <span className="text-white">{stats.frameCount}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-400">Frame Rate</span>
                  <span className="text-white">{frameRate} fps</span>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Settings Modal */}
      {showSettings && (
        <div className="fixed inset-0 bg-black/60 backdrop-blur-sm flex items-center justify-center p-4 z-50">
          <div className="bg-slate-900 border border-purple-500/20 rounded-2xl p-6 max-w-md w-full">
            <h2 className="text-xl font-bold mb-4">Settings</h2>

            <div className="space-y-4">
              <div>
                <label htmlFor="backend-url" className="block text-sm text-gray-400 mb-2">
                  Backend URL
                </label>
                <input
                  id="backend-url"
                  type="text"
                  value={backendUrl}
                  onChange={(e) => setBackendUrl(e.target.value)}
                  className="w-full px-4 py-2 bg-black/40 border border-purple-500/20 rounded-lg focus:outline-none focus:border-purple-500 text-white"
                  placeholder="ws://localhost:8000"
                />
                <p className="text-xs text-gray-500 mt-1">WebSocket URL (without /ws/...)</p>
              </div>

              <div>
                <label htmlFor="frame-rate" className="block text-sm text-gray-400 mb-2">
                  Frame Rate (fps)
                </label>
                <select
                  id="frame-rate"
                  value={frameRate}
                  onChange={(e) => setFrameRate(Number(e.target.value))}
                  className="w-full px-4 py-2 bg-black/40 border border-purple-500/20 rounded-lg focus:outline-none focus:border-purple-500 text-white"
                >
                  <option value={5}>5 fps (Low CPU)</option>
                  <option value={10}>10 fps (Balanced)</option>
                  <option value={15}>15 fps (Smooth)</option>
                  <option value={30}>30 fps (High Quality)</option>
                </select>
              </div>

              <div className="flex items-center justify-between">
                <span className="text-sm">Enable Audio Feedback</span>
                <button
                  onClick={() => setIsAudioEnabled(!isAudioEnabled)}
                  className={`relative w-12 h-6 rounded-full transition ${isAudioEnabled ? 'bg-purple-600' : 'bg-gray-600'
                    }`}
                  type="button"
                  aria-label="Toggle audio feedback"
                >
                  <div
                    className={`absolute top-0.5 left-0.5 w-5 h-5 bg-white rounded-full transition-transform ${isAudioEnabled ? 'translate-x-6' : 'translate-x-0'
                      }`}
                  />
                </button>
              </div>

              <div className="pt-4 border-t border-purple-500/20">
                <h3 className="text-sm font-semibold mb-2">Connection Info</h3>
                <div className="text-xs text-gray-400 space-y-1">
                  <p>• Video: {backendUrl}/ws/video-analysis</p>
                  <p>• Audio: {backendUrl}/ws/coach-audio</p>
                  <p>• Status: {connectionStatus}</p>
                </div>
              </div>
            </div>

            <button
              onClick={() => setShowSettings(false)}
              className="w-full mt-6 px-4 py-2 bg-purple-600 hover:bg-purple-700 rounded-lg transition"
              type="button"
            >
              Close Settings
            </button>
          </div>
        </div>
      )}
    </div>
  );
};

export default VideoCoachApp;