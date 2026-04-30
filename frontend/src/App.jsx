import React from 'react';
import { BrowserRouter, Routes, Route } from 'react-router-dom';
import { ThemeProvider } from './context/ThemeContext';
import { DatabaseProvider } from './context/DatabaseContext';
import { ToastProvider } from './context/ToastContext';
import { initTokenCleanup } from './utils/tokenCleanup';
import ErrorBoundary from './components/ErrorBoundary';
import Toast from './components/Toast';
import Layout from './components/Layout';
import Dashboard from './pages/Dashboard';
import Statistics from './pages/Statistics';
import Tournament from './pages/Tournament';
import NewGame from './pages/NewGame';
import Arena from './pages/Arena';
import Admin from './pages/Admin';
import History from './pages/History';
import Settings from './pages/Settings';

// Per-route boundary helper — keep one instance per page so a crash in one
// route doesn't propagate up past the navigation chrome.
const Bounded = (element) => <ErrorBoundary>{element}</ErrorBoundary>;

function App() {
  // Initialize token cleanup on app startup
  React.useEffect(() => {
    initTokenCleanup();
  }, []);

  return (
    <ToastProvider>
      <DatabaseProvider>
        <ThemeProvider>
          <BrowserRouter
            future={{ v7_startTransition: true, v7_relativeSplatPath: true }}
          >
            <Layout>
              <Routes>
                <Route path="/" element={Bounded(<Dashboard />)} />
                <Route path="/tournament" element={Bounded(<Tournament />)} />
                <Route path="/statistics" element={Bounded(<Statistics />)} />
                <Route path="/new" element={Bounded(<NewGame />)} />
                <Route path="/game/:id" element={Bounded(<Arena />)} />
                <Route path="/history" element={Bounded(<History />)} />
                <Route path="/settings" element={Bounded(<Settings />)} />
                <Route path="/admin" element={Bounded(<Admin />)} />
              </Routes>
            </Layout>
            <Toast />
          </BrowserRouter>
        </ThemeProvider>
      </DatabaseProvider>
    </ToastProvider>
  );
}

export default App;