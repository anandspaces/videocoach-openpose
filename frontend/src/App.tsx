import React, { useState } from 'react';
import HomePage from './pages/HomePage';
import MeetingPage from './pages/MeetingPage';

const App: React.FC = () => {
  const [currentRoute, setCurrentRoute] = useState<string>('home');

  const navigate = (path: string) => {
    setCurrentRoute(path);
  };

  const renderPage = () => {
    if (currentRoute === 'home' || currentRoute === '/') {
      return <HomePage onNavigate={navigate} />;
    }

    if (currentRoute.startsWith('/meet/')) {
      const sessionId = currentRoute.replace('/meet/', '');
      return <MeetingPage sessionId={sessionId} onNavigate={navigate} />;
    }

    return <HomePage onNavigate={navigate} />;
  };

  return <>{renderPage()}</>;
};

export default App;