import React from 'react';

import { DatabaseProvider } from '../context/DatabaseContext';
import { ThemeProvider } from '../context/ThemeContext';
import { ToastProvider } from '../context/ToastContext';

/**
 * Standard provider stack used in component / hook tests. ToastProvider must
 * wrap DatabaseProvider because DatabaseContext's response interceptor calls
 * useToast() during mount.
 */
export const AllProviders = ({ children }) => (
  <ToastProvider>
    <DatabaseProvider>
      <ThemeProvider>{children}</ThemeProvider>
    </DatabaseProvider>
  </ToastProvider>
);

export const dbWrapper = ({ children }) => <AllProviders>{children}</AllProviders>;
