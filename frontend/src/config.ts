// ============================================================================
// CONFIGURATION
// ============================================================================

export const BACKEND_URL = import.meta.env.VITE_MEET_BASE_URL;
export const WS_BASE_URL = import.meta.env.VITE_WEBSOCKET_BASE_URL;

export const POSE_PAIRS = [
  [0, 1], [1, 2], [2, 3], [3, 4],
  [1, 5], [5, 6], [6, 7],
  [1, 8], [8, 9], [9, 10],
  [1, 11], [11, 12], [12, 13],
  [0, 14], [0, 15], [14, 16], [15, 17]
];