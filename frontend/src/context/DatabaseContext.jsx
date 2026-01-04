import { createContext, useContext, useState, useEffect, useRef } from 'react';
import { apiClient } from '../api/client';

const DatabaseContext = createContext();

export const DatabaseProvider = ({ children }) => {
  // Persist choice in LocalStorage
  const [dbEnv, setDbEnv] = useState(() => localStorage.getItem('dbEnv') || 'prod');

  // Use a Ref to hold the latest env value.
  // This solves the race condition where child components fetch data 
  // before the parent's useEffect has a chance to update the interceptor.
  const dbEnvRef = useRef(dbEnv);
  
  // Update the ref synchronously during render
  dbEnvRef.current = dbEnv;

  useEffect(() => {
    localStorage.setItem('dbEnv', dbEnv);
  }, [dbEnv]);

  // Register the interceptor ONLY ONCE on mount.
  // It reads the dynamic value from dbEnvRef.current.
  useEffect(() => {
    const interceptor = apiClient.interceptors.request.use((config) => {
      config.headers['x-db-env'] = dbEnvRef.current;
      return config;
    });

    return () => apiClient.interceptors.request.eject(interceptor);
  }, []); // Empty dependency array ensures this doesn't reset causing timing issues

  const toggleDb = () => {
    setDbEnv(prev => prev === 'prod' ? 'test' : 'prod');
  };

  return (
    <DatabaseContext.Provider value={{ dbEnv, toggleDb }}>
      {children}
    </DatabaseContext.Provider>
  );
};

export const useDatabase = () => useContext(DatabaseContext);