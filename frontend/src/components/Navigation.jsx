import React from 'react';
import { Link, useLocation } from 'react-router-dom';
import { useTheme } from '../context/ThemeContext';
import { useDatabase } from '../context/DatabaseContext';
import { Sun, Moon, LayoutGrid, PlusCircle, History as HistoryIcon, Shield, BarChart2, Trophy } from 'lucide-react';

const Navigation = () => {
  const location = useLocation();
  const { theme, toggleTheme } = useTheme();
  const { dbEnv, toggleDb } = useDatabase();

  const isActive = (path) => location.pathname === path;

  const NavLink = ({ to, icon: Icon, label }) => (
    <Link
      to={to}
      className={`flex items-center gap-2 px-3 py-2 rounded-md text-sm font-medium transition-colors ${
        isActive(to)
          ? 'bg-gray-200 dark:bg-gray-800 text-gray-900 dark:text-white'
          : 'text-gray-600 dark:text-gray-400 hover:bg-gray-100 dark:hover:bg-gray-800'
      }`}
    >
      <Icon size={18} />
      <span>{label}</span>
    </Link>
  );

  return (
    <nav className="sticky top-0 z-50 w-full border-b border-gray-200 dark:border-gray-800 bg-white/80 dark:bg-gray-900/80 backdrop-blur-md">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex justify-between items-center h-16">
          {/* Logo */}
          <Link to="/" className="flex items-center gap-2">
            <div className="w-8 h-8 bg-brand-600 rounded-lg flex items-center justify-center text-white font-bold text-lg">
              C4
            </div>
            <span className="font-bold text-xl tracking-tight text-gray-900 dark:text-white hidden sm:block">
              Connect<span className="text-brand-600">AI</span>
            </span>
          </Link>

          {/* Links */}
          <div className="hidden md:flex items-center space-x-1">
            <NavLink to="/" icon={LayoutGrid} label="Overview" />
            <NavLink to="/tournament" icon={Trophy} label="Tournament" />
            <NavLink to="/statistics" icon={BarChart2} label="Statistics" />
            <NavLink to="/new" icon={PlusCircle} label="New Match" />
            <NavLink to="/history" icon={HistoryIcon} label="History" />
            <NavLink to="/admin" icon={Shield} label="Admin" />
          </div>

          {/* Database Environment Toggle */}
          <button
            onClick={toggleDb}
            className="px-3 py-1 rounded-full text-sm font-medium transition-colors focus:outline-none mr-2"
            aria-label="Toggle Database Environment"
            style={{
              backgroundColor: dbEnv === 'prod' ? '#10B981' : '#F59E0B',
              color: 'white'
            }}
          >
            {dbEnv === 'prod' ? 'Production' : 'Test Sandbox'}
          </button>
          
          {/* Theme Toggle */}
          <button
            onClick={toggleTheme}
            className="p-2 rounded-full text-gray-500 hover:bg-gray-100 dark:hover:bg-gray-800 transition-colors focus:outline-none"
            aria-label="Toggle Theme"
          >
            {theme === 'light' ? (
              <Moon size={20} className="text-gray-700" />
            ) : (
              <Sun size={20} className="text-yellow-400" />
            )}
          </button>
        </div>
      </div>
    </nav>
  );
};

export default Navigation;