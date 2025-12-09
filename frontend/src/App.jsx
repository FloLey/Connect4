import { BrowserRouter, Routes, Route } from 'react-router-dom';
import { ThemeProvider } from './context/ThemeContext';
import Layout from './components/Layout';
import Dashboard from './pages/Dashboard';
import Statistics from './pages/Statistics';
import Tournament from './pages/Tournament';
import NewGame from './pages/NewGame';
import Arena from './pages/Arena';
import Admin from './pages/Admin';
import History from './pages/History';

function App() {
  return (
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
  );
}

export default App;