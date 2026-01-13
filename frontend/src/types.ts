// ============================================================================
// TYPE DEFINITIONS
// ============================================================================

export interface KeyPoint {
  x: number;
  y: number;
  confidence: number;
}

export interface AnalysisData {
  frame_num: number;
  timestamp: number;
  keypoints: (KeyPoint | null)[];
  joints: Record<string, number>;
  symmetry: Record<string, number>;
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
    all_emotions?: Record<string, number>;
  };
  posture: {
    status: string;
    angle: number;
    shoulder_aligned?: boolean;
  };
  activities: string[];
}

export interface CoachingFeedback {
  triggered: boolean;
  reason: string;
  feedback: string;
}

export interface WebSocketMessage {
  type: 'welcome' | 'analysis' | 'error' | 'pong';
  message?: string;
  data?: AnalysisData;
  coaching?: CoachingFeedback;
  session_id?: string;
  participant_id?: string;
}

export interface Stats {
  frameCount: number;
  balance: number;
  energy: string;
  emotion: string;
  emotionConfidence: number;
  posture: string;
  postureAngle: number;
}

export interface FeedbackItem {
  time: string;
  text: string;
  reason?: string;
}

export type ConnectionStatus = 'disconnected' | 'connected' | 'error' | 'connecting';

export interface MeetingSession {
  session_id: string;
  meeting_link: string;
  ws_endpoint: string;
  created_at: string;
  host_id?: string;
  expires_at?: string;
}