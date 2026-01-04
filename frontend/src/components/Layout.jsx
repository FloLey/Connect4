import React from 'react';
import Navigation from './Navigation';
import { useDatabase } from '../context/DatabaseContext';

const Layout = ({ children }) => {
  const { dbEnv } = useDatabase();
  
  return (
    <div className="min-h-screen transition-colors duration-300 ease-in-out bg-gray-50 dark:bg-gray-950 text-gray-900 dark:text-gray-100">
      <Navigation />
      {dbEnv === 'test' && (
        <div className="bg-yellow-500 text-black text-center py-2 px-4">
          ⚠️ TEST ENVIRONMENT: You are using the sandbox database. API costs are NOT isolated - any LLM usage will charge your OpenAI/Anthropic account.
        </div>
      )}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {children}
      </main>
    </div>
  );
};

export default Layout;