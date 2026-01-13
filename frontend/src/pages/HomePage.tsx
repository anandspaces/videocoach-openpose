import React, { useState } from 'react';
import { Activity, Video, Users } from 'lucide-react';
import { BACKEND_URL } from '../config';

interface HomePageProps {
  onNavigate: (path: string) => void;
}

const HomePage: React.FC<HomePageProps> = ({ onNavigate }) => {
  const [isCreating, setIsCreating] = useState(false);
  const [joinSessionId, setJoinSessionId] = useState('');
  // const [recentSessions, setRecentSessions] = useState<MeetingSession[]>([]);

  // useEffect(() => {
  //   const saved = localStorage.getItem('recent_sessions');
  //   if (saved) {
  //     try {
  //       setRecentSessions(JSON.parse(saved));
  //     } catch (e) {
  //       console.error('Failed to load recent sessions:', e);
  //     }
  //   }
  // }, []);

  const createMeeting = async () => {
    setIsCreating(true);
    try {
      const response = await fetch(`${BACKEND_URL}/api/create-meeting`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' }
      });

      if (!response.ok) throw new Error('Failed to create meeting');

      const data = await response.json();

      // const newSession: MeetingSession = {
      //   session_id: data.session_id,
      //   meeting_link: data.meeting_link,
      //   ws_endpoint: data.ws_endpoint,
      //   created_at: new Date().toISOString()
      // };

      // const updated = [newSession, ...recentSessions.slice(0, 4)];
      // setRecentSessions(updated);
      // localStorage.setItem('recent_sessions', JSON.stringify(updated));

      onNavigate(`/meet/${data.session_id}`);
    } catch (error) {
      console.error('Error creating meeting:', error);
      alert('Failed to create meeting. Please check if backend is running.');
    } finally {
      setIsCreating(false);
    }
  };

  const joinMeeting = () => {
    if (joinSessionId.trim()) {
      onNavigate(`/meet/${joinSessionId.trim()}`);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900 text-white">
      <div className="max-w-4xl mx-auto px-4 py-16">
        <div className="text-center mb-12">
          <div className="inline-flex items-center gap-3 mb-4">
            <div className="bg-purple-600 p-4 rounded-2xl">
              <Activity className="w-12 h-12" />
            </div>
          </div>
          <h1 className="text-5xl font-bold mb-4 bg-gradient-to-r from-purple-400 to-pink-400 bg-clip-text text-transparent">
            AI Video Coach
          </h1>
          <p className="text-xl text-gray-300">Real-time Pose Analysis & Coaching</p>
        </div>

        <div className="grid md:grid-cols-2 gap-6 mb-12">
          <div className="bg-black/40 backdrop-blur-sm rounded-2xl border border-purple-500/20 p-8 hover:border-purple-500/40 transition">
            <div className="bg-purple-600/20 p-4 rounded-xl w-fit mb-4">
              <Video className="w-8 h-8 text-purple-400" />
            </div>
            <h2 className="text-2xl font-bold mb-2">Create Meeting</h2>
            <p className="text-gray-400 mb-6">Start a new AI coaching session</p>
            <button
              onClick={createMeeting}
              disabled={isCreating}
              className="w-full py-3 bg-purple-600 hover:bg-purple-700 disabled:bg-gray-600 rounded-xl font-semibold transition flex items-center justify-center gap-2"
            >
              {isCreating ? (
                <>
                  <div className="w-5 h-5 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                  Creating...
                </>
              ) : (
                <>
                  <Video className="w-5 h-5" />
                  New Meeting
                </>
              )}
            </button>
          </div>

          <div className="bg-black/40 backdrop-blur-sm rounded-2xl border border-purple-500/20 p-8 hover:border-purple-500/40 transition">
            <div className="bg-green-600/20 p-4 rounded-xl w-fit mb-4">
              <Users className="w-8 h-8 text-green-400" />
            </div>
            <h2 className="text-2xl font-bold mb-2">Join Meeting</h2>
            <p className="text-gray-400 mb-6">Enter a session ID to join</p>
            <div className="space-y-3">
              <input
                type="text"
                value={joinSessionId}
                onChange={(e) => setJoinSessionId(e.target.value)}
                onKeyPress={(e) => e.key === 'Enter' && joinMeeting()}
                placeholder="Enter session ID"
                className="w-full px-4 py-3 bg-black/40 border border-purple-500/20 rounded-xl focus:outline-none focus:border-purple-500 text-white"
              />
              <button
                onClick={joinMeeting}
                disabled={!joinSessionId.trim()}
                className="w-full py-3 bg-green-600 hover:bg-green-700 disabled:bg-gray-600 rounded-xl font-semibold transition flex items-center justify-center gap-2"
              >
                <Users className="w-5 h-5" />
                Join Session
              </button>
            </div>
          </div>
        </div>

        {/* {recentSessions.length > 0 && (
          <div className="bg-black/40 backdrop-blur-sm rounded-2xl border border-purple-500/20 p-6">
            <h3 className="text-lg font-bold mb-4 flex items-center gap-2">
              <LinkIcon className="w-5 h-5" />
              Recent Sessions
            </h3>
            <div className="space-y-2">
              {recentSessions.map((session) => (
                <button
                  key={session.session_id}
                  onClick={() => onNavigate(`/meet/${session.session_id}`)}
                  className="w-full p-4 bg-purple-500/10 hover:bg-purple-500/20 border border-purple-500/20 rounded-xl transition text-left"
                >
                  <div className="flex items-center justify-between">
                    <div>
                      <div className="font-mono text-sm text-purple-300">
                        {session.session_id.slice(0, 8)}...
                      </div>
                      <div className="text-xs text-gray-400 mt-1">
                        {new Date(session.created_at).toLocaleString()}
                      </div>
                    </div>
                    <ArrowLeft className="w-5 h-5 rotate-180" />
                  </div>
                </button>
              ))}
            </div>
          </div>
        )} */}
      </div>
    </div>
  );
};

export default HomePage;