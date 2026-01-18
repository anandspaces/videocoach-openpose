import React, { useState } from 'react';
import { Activity, Video } from 'lucide-react';
import { BACKEND_URL } from '../config';

interface HomePageProps {
  onNavigate: (path: string) => void;
}

const HomePage: React.FC<HomePageProps> = ({ onNavigate }) => {
  const [isCreating, setIsCreating] = useState(false);
  // const [joinSessionId, setJoinSessionId] = useState('');

  const createMeeting = async () => {
    setIsCreating(true);
    try {
      const response = await fetch(`${BACKEND_URL}/api/create-meeting`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' }
      });

      if (!response.ok) throw new Error('Failed to create meeting');

      const data = await response.json();

      onNavigate(`/meet/${data.session_id}`);
    } catch (error) {
      console.error('Error creating meeting:', error);
      alert('Failed to create meeting. Please check if backend is running.');
    } finally {
      setIsCreating(false);
    }
  };

  // const joinMeeting = () => {
  //   if (joinSessionId.trim()) {
  //     onNavigate(`/meet/${joinSessionId.trim()}`);
  //   }
  // };

  return (
    <div className="h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900 text-white overflow-hidden">
      <div className="h-full max-w-4xl mx-auto px-4 flex flex-col justify-center">
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

        <div className="max-w-md mx-auto">
          <div className="bg-black/40 backdrop-blur-sm rounded-2xl border border-purple-500/20 p-8 hover:border-purple-500/40 transition">
            <div className="bg-purple-600/20 p-4 rounded-xl w-fit mb-4 mx-auto">
              <Video className="w-8 h-8 text-purple-400" />
            </div>
            <h2 className="text-2xl font-bold mb-2 text-center">Create Meeting</h2>
            <p className="text-gray-400 mb-6 text-center">Start a new AI coaching session</p>
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
        </div>

        {/* <div className="bg-black/40 backdrop-blur-sm rounded-2xl border border-purple-500/20 p-8 hover:border-purple-500/40 transition">
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
        </div> */}
      </div>
    </div>
  );
};

export default HomePage;