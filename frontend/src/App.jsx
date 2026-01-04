import React from 'react';
import { BrowserRouter, Routes, Route } from 'react-router-dom';
import { ThemeProvider } from './context/ThemeContext';
import { DatabaseProvider } from './context/DatabaseContext';
import { initTokenCleanup } from './utils/tokenCleanup';
import Layout from './components/Layout';
import Dashboard from './pages/Dashboard';
import Statistics from './pages/Statistics';
import Tournament from './pages/Tournament';
import NewGame from './pages/NewGame';
import Arena from './pages/Arena';
import Admin from './pages/Admin';
import History from './pages/History';

function App() {
  // Initialize token cleanup on app startup
  React.useEffect(() => {
    initTokenCleanup();
  }, []);

  return (
    <DatabaseProvider>
      <ThemeProvider>
        <BrowserRouter>
          <Layout>
            <Routes>
              <Route path="/" element={<Dashboard />} />
              <Route path="/tournament" element={<Tournament />} />
              <Route path="/statistics" element={<Statistics />} />
              <Route path="/new" element={<NewGame />} />
              <Route path="/game/:id" element={<Arena />} />
              <Route path="/history" element={<History />} />
              <Route path="/admin" element={<Admin />} />
            </Routes>
          </Layout>
        </BrowserRouter>
      </ThemeProvider>
    </DatabaseProvider>
  );
}

export default App;