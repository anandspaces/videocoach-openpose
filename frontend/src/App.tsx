import React, { useState, useEffect, useRef, useCallback } from 'react';
import { Camera, Activity, Heart, Zap, Volume2, VolumeX, Play, Pause, Settings } from 'lucide-react';

// TypeScript Interfaces
interface KeyPoint {
  x: number;
  y: number;
  confidence: number;
}

interface JointAngles {
  right_elbow: number;
  left_elbow: number;
  right_knee: number;
  left_knee: number;
  right_hip?: number;
  left_hip?: number;
}

interface SymmetryData {
  shoulder_width: number;
  arm_symmetry: number;
  leg_symmetry: number;
  hip_width?: number;
}

interface BalanceData {
  cog: [number, number];
  balance_score: number;
}

interface MovementData {
  energy: string;
  movement_score: number;
  velocity: number;
  sentiment?: string;
}

interface EmotionData {
  emotion: string;
  confidence: number;
  sentiment: string;
  details?: string;
}

interface PostureData {
  status: string;
  angle: number;
  color?: [number, number, number];
  shoulder_aligned?: boolean;
}

interface PoseFrameData {
  frame_num: number;
  timestamp: number;
  keypoints: (KeyPoint | null)[];
  joints: JointAngles;
  symmetry: SymmetryData;
  balance: BalanceData;
  movement: MovementData;
  emotion: EmotionData;
  posture: PostureData;
  activities: string[];
}

interface Stats {
  frameCount: number;
  balance: number;
  energy: string;
  emotion: string;
  emotionConfidence: number;
  posture: string;
}

interface FeedbackItem {
  time: string;
  text: string;
  reason?: string;
}

type ConnectionStatus = 'disconnected' | 'connected' | 'error' | 'connecting';

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
    posture: 'Unknown'
  });
  const [recentFeedback, setRecentFeedback] = useState<FeedbackItem[]>([]);
  const [showSettings, setShowSettings] = useState<boolean>(false);
  const [backendUrl, setBackendUrl] = useState<string>('ws://localhost:8000');

  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const wsVideoRef = useRef<WebSocket | null>(null);
  const wsAudioRef = useRef<WebSocket | null>(null);
  const audioContextRef = useRef<AudioContext | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const animationFrameRef = useRef<number | null>(null);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (wsVideoRef.current) wsVideoRef.current.close();
      if (wsAudioRef.current) wsAudioRef.current.close();
      if (streamRef.current) {
        streamRef.current.getTracks().forEach(track => track.stop());
      }
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }
      if (audioContextRef.current) {
        audioContextRef.current.close();
      }
    };
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
        console.log('📥 Backend response:', response);
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

            // Add feedback to UI
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
  }, [backendUrl, isAudioEnabled]);

  const startStreaming = async (): Promise<void> => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: 640, height: 480 }
      });

      streamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
      }

      connectWebSockets();
      setIsStreaming(true);

      // Start frame processing
      processFrames();
    } catch (err) {
      console.error('Camera access error:', err);
      alert('Could not access camera. Please grant permission.');
    }
  };

  const stopStreaming = (): void => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => track.stop());
    }
    if (wsVideoRef.current) wsVideoRef.current.close();
    if (wsAudioRef.current) wsAudioRef.current.close();
    if (animationFrameRef.current) {
      cancelAnimationFrame(animationFrameRef.current);
    }

    setIsStreaming(false);
    setConnectionStatus('disconnected');
  };

  const generateMockPoseData = useCallback((): PoseFrameData => {
    const emotions: string[] = ['Happy', 'Neutral', 'Focused', 'Tired'];
    const energyLevels: string[] = ['Low (Calm)', 'Medium (Active)', 'High (Moving)'];
    const postureStates: string[] = ['Excellent', 'Good', 'Fair', 'Poor'];

    return {
      frame_num: stats.frameCount + 1,
      timestamp: Date.now() / 1000,
      keypoints: Array(18).fill(null).map((_) =>
        Math.random() > 0.3 ? {
          x: Math.random() * 640,
          y: Math.random() * 480,
          confidence: Math.random()
        } : null
      ),
      joints: {
        right_elbow: 120 + Math.random() * 60,
        left_elbow: 120 + Math.random() * 60,
        right_knee: 140 + Math.random() * 40,
        left_knee: 140 + Math.random() * 40
      },
      symmetry: {
        shoulder_width: 300 + Math.random() * 50,
        arm_symmetry: Math.random() * 15,
        leg_symmetry: Math.random() * 15
      },
      balance: {
        cog: [320, 240],
        balance_score: 40 + Math.random() * 50
      },
      movement: {
        energy: energyLevels[Math.floor(Math.random() * energyLevels.length)],
        movement_score: Math.random() * 30,
        velocity: Math.random() * 40
      },
      emotion: {
        emotion: emotions[Math.floor(Math.random() * emotions.length)],
        confidence: 60 + Math.random() * 30,
        sentiment: 'Positive'
      },
      posture: {
        status: postureStates[Math.floor(Math.random() * postureStates.length)],
        angle: Math.random() * 50
      },
      activities: ['Normal Pose']
    };
  }, [stats.frameCount]);

  const processFrames = useCallback((): void => {
    const canvas = canvasRef.current;
    const video = videoRef.current;

    if (!canvas || !video) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const processFrame = (): void => {
      if (!isStreaming) return;

      // Draw video frame to canvas
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

      // Generate mock pose data (replace with actual OpenPose integration)
      const mockData = generateMockPoseData();

      // Update UI stats
      setStats(prev => ({
        frameCount: prev.frameCount + 1,
        balance: mockData.balance.balance_score,
        energy: mockData.movement.energy,
        emotion: mockData.emotion.emotion,
        emotionConfidence: mockData.emotion.confidence,
        posture: mockData.posture.status
      }));

      // Send to backend
      if (wsVideoRef.current?.readyState === WebSocket.OPEN) {
        wsVideoRef.current.send(JSON.stringify(mockData));
      }

      // Continue processing
      animationFrameRef.current = requestAnimationFrame(processFrame);
    };

    processFrame();
  }, [isStreaming, generateMockPoseData]);

  useEffect(() => {
    if (isStreaming) {
      processFrames();
    }
  }, [isStreaming, processFrames]);

  const getPostureColorClass = (posture: string): string => {
    switch (posture) {
      case 'Excellent': return 'bg-green-500/20 border border-green-500/30';
      case 'Good': return 'bg-blue-500/20 border border-blue-500/30';
      case 'Fair': return 'bg-yellow-500/20 border border-yellow-500/30';
      default: return 'bg-red-500/20 border border-red-500/30';
    }
  };

  const getConnectionStatusColor = (status: ConnectionStatus): string => {
    switch (status) {
      case 'connected': return 'bg-green-500/20 text-green-300';
      case 'error': return 'bg-red-500/20 text-red-300';
      case 'connecting': return 'bg-yellow-500/20 text-yellow-300';
      default: return 'bg-gray-500/20 text-gray-300';
    }
  };

  const getConnectionDotColor = (status: ConnectionStatus): string => {
    switch (status) {
      case 'connected': return 'bg-green-400 animate-pulse';
      case 'error': return 'bg-red-400';
      case 'connecting': return 'bg-yellow-400 animate-pulse';
      default: return 'bg-gray-400';
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
            <div className={`flex items-center gap-2 px-3 py-1 rounded-full text-sm ${getConnectionStatusColor(connectionStatus)}`}>
              <div className={`w-2 h-2 rounded-full ${getConnectionDotColor(connectionStatus)}`} />
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
                  className="w-full h-full object-cover"
                />
                <canvas
                  ref={canvasRef}
                  width={640}
                  height={480}
                  className="absolute inset-0 w-full h-full opacity-0"
                />

                {!isStreaming && (
                  <div className="absolute inset-0 flex items-center justify-center bg-black/60">
                    <div className="text-center">
                      <Camera className="w-16 h-16 mx-auto mb-4 text-purple-400" />
                      <p className="text-lg mb-2">Camera Ready</p>
                      <p className="text-sm text-gray-400">Click Start to begin analysis</p>
                    </div>
                  </div>
                )}

                {/* Live Stats Overlay */}
                {isStreaming && (
                  <div className="absolute top-4 left-4 space-y-2">
                    <div className="bg-black/70 backdrop-blur-sm px-3 py-2 rounded-lg">
                      <div className="text-xs text-purple-300">Balance</div>
                      <div className="text-lg font-bold">{stats.balance.toFixed(0)}/100</div>
                    </div>
                    <div className="bg-black/70 backdrop-blur-sm px-3 py-2 rounded-lg">
                      <div className="text-xs text-purple-300">Emotion</div>
                      <div className="text-sm font-bold">{stats.emotion}</div>
                    </div>
                  </div>
                )}

                {/* Frame Counter */}
                {isStreaming && (
                  <div className="absolute top-4 right-4 bg-black/70 backdrop-blur-sm px-3 py-2 rounded-lg">
                    <div className="text-xs text-purple-300">Frame</div>
                    <div className="text-lg font-bold">{stats.frameCount}</div>
                  </div>
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
                    <span className="text-gray-400">Energy</span>
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
                  <div className={`px-3 py-2 rounded-lg text-sm ${getPostureColorClass(stats.posture)}`}>
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
              <h2 className="text-lg font-bold mb-4">Recent Feedback</h2>

              {recentFeedback.length === 0 ? (
                <div className="text-center py-8 text-gray-400">
                  <Volume2 className="w-8 h-8 mx-auto mb-2 opacity-50" />
                  <p className="text-sm">No feedback yet</p>
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
                    </div>
                  ))}
                </div>
              )}
            </div>

            {/* Connection Info */}
            <div className="bg-black/40 backdrop-blur-sm rounded-2xl border border-purple-500/20 p-6">
              <h2 className="text-lg font-bold mb-4">System Status</h2>
              <div className="space-y-2 text-sm">
                <div className="flex justify-between">
                  <span className="text-gray-400">Backend</span>
                  <span className={connectionStatus === 'connected' ? 'text-green-400' : 'text-red-400'}>
                    {connectionStatus === 'connected' ? '✓ Connected' : '✗ Disconnected'}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-400">Audio</span>
                  <span className={isAudioEnabled ? 'text-green-400' : 'text-gray-400'}>
                    {isAudioEnabled ? '✓ Enabled' : '✗ Disabled'}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-400">Frames</span>
                  <span className="text-white">{stats.frameCount}</span>
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
                  className="w-full px-4 py-2 bg-black/40 border border-purple-500/20 rounded-lg focus:outline-none focus:border-purple-500"
                />
              </div>

              <div>
                <label htmlFor="camera-resolution" className="block text-sm text-gray-400 mb-2">
                  Camera Resolution
                </label>
                <select
                  id="camera-resolution"
                  className="w-full px-4 py-2 bg-black/40 border border-purple-500/20 rounded-lg focus:outline-none focus:border-purple-500"
                >
                  <option>640x480</option>
                  <option>1280x720</option>
                  <option>1920x1080</option>
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
                  <div className={`absolute top-0.5 left-0.5 w-5 h-5 bg-white rounded-full transition-transform ${isAudioEnabled ? 'translate-x-6' : 'translate-x-0'
                    }`} />
                </button>
              </div>
            </div>

            <button
              onClick={() => setShowSettings(false)}
              className="w-full mt-6 px-4 py-2 bg-purple-600 hover:bg-purple-700 rounded-lg transition"
              type="button"
            >
              Close
            </button>
          </div>
        </div>
      )}
    </div>
  );
};

export default VideoCoachApp;