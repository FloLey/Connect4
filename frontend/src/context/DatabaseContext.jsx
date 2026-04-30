import { createContext, useContext, useState, useEffect, useRef } from 'react';
import { apiClient } from '../api/client';
import { useToast } from './ToastContext';

const DatabaseContext = createContext();

export const DatabaseProvider = ({ children }) => {
  // Persist choice in LocalStorage
  const [dbEnv, setDbEnv] = useState(() => localStorage.getItem('dbEnv') || 'prod');
  const toast = useToast();

  // Use a Ref to hold the latest env value.
  // This solves the race condition where child components fetch data
  // before the parent's useEffect has a chance to update the interceptor.
  const dbEnvRef = useRef(dbEnv);
  dbEnvRef.current = dbEnv;

  // Toast handler is read through a ref so the response interceptor (registered
  // once on mount) always uses the current implementation.
  const toastRef = useRef(toast);
  toastRef.current = toast;

  useEffect(() => {
    localStorage.setItem('dbEnv', dbEnv);
  }, [dbEnv]);

  // Register the interceptors ONLY ONCE on mount.
  useEffect(() => {
    const requestInterceptor = apiClient.interceptors.request.use((config) => {
      config.headers['x-db-env'] = dbEnvRef.current;
      // Admin token (only relevant for /admin/* routes; backend ignores the
      // header on others). We read it from localStorage here so token rotation
      // doesn't require an app reload.
      const adminToken = localStorage.getItem('admin_token');
      if (adminToken) {
        config.headers['X-Admin-Token'] = adminToken;
      }
      return config;
    });

    const responseInterceptor = apiClient.interceptors.response.use(
      (response) => response,
      (error) => {
        // Surface 4xx/5xx + network failures as toasts. Pages can still attach
        // their own .catch handlers — this is the catch-all so failures stop
        // disappearing into the void.
        const status = error?.response?.status;
        const detail =
          error?.response?.data?.detail ||
          error?.response?.data?.message ||
          error?.message;
        const summary = status ? `${status} ${detail || ''}`.trim() : detail || 'Request failed';
        toastRef.current?.error(summary, { durationMs: 6000 });
        return Promise.reject(error);
      }
    );

    return () => {
      apiClient.interceptors.request.eject(requestInterceptor);
      apiClient.interceptors.response.eject(responseInterceptor);
    };
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