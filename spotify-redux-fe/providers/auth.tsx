import React, { createContext, useContext, useEffect, useMemo, useState } from 'react';
import {
  getItemAsync as secureGet,
  setItemAsync as secureSet,
  deleteItemAsync as secureDelete,
  isAvailableAsync as secureAvailable,
} from 'expo-secure-store';

import { login as apiLogin, signup as apiSignup } from '@/lib/api';

type AuthContextValue = {
  token: string | null;
  username: string | null;
  ready: boolean;
  login: (username: string, password: string) => Promise<void>;
  signup: (username: string, password: string) => Promise<void>;
  logout: () => void;
};

const AuthContext = createContext<AuthContextValue | undefined>(undefined);

export function AuthProvider({ children }: { children: React.ReactNode }) {
  const [token, setToken] = useState<string | null>(null);
  const [username, setUsername] = useState<string | null>(null);
  const [ready, setReady] = useState(false);

  const storage = useMemo(() => {
    const hasSecure = typeof secureAvailable === 'function';
    return {
      get: async (key: string) => {
        try {
          if (hasSecure && (await secureAvailable())) {
            return await secureGet(key);
          }
          if (typeof window !== 'undefined' && window.localStorage) {
            return window.localStorage.getItem(key);
          }
        } catch {
          // ignore and fallback
        }
        return null;
      },
      set: async (key: string, value: string) => {
        try {
          if (hasSecure && (await secureAvailable())) {
            await secureSet(key, value);
            return;
          }
          if (typeof window !== 'undefined' && window.localStorage) {
            window.localStorage.setItem(key, value);
          }
        } catch {
          // ignore
        }
      },
      del: async (key: string) => {
        try {
          if (hasSecure && (await secureAvailable())) {
            await secureDelete(key);
            return;
          }
          if (typeof window !== 'undefined' && window.localStorage) {
            window.localStorage.removeItem(key);
          }
        } catch {
          // ignore
        }
      },
    };
  }, []);

  // Load persisted creds
  useEffect(() => {
    (async () => {
      try {
        const storedToken = await storage.get('auth_token');
        const storedUser = await storage.get('auth_username');
        if (storedToken) setToken(storedToken);
        if (storedUser) setUsername(storedUser);
      } finally {
        setReady(true);
      }
    })();
  }, [storage]);

  const login = async (user: string, password: string) => {
    const res = await apiLogin(user, password);
    setToken(res.access_token);
    setUsername(user);
    await storage.set('auth_token', res.access_token);
    await storage.set('auth_username', user);
  };

  const signup = async (user: string, password: string) => {
    const res = await apiSignup(user, password);
    setToken(res.access_token);
    setUsername(user);
    await storage.set('auth_token', res.access_token);
    await storage.set('auth_username', user);
  };

  const logout = () => {
    setToken(null);
    setUsername(null);
    storage.del('auth_token');
    storage.del('auth_username');
  };

  const value = useMemo(
    () => ({
      token,
      username,
      ready,
      login,
      signup,
      logout,
    }),
    [token, username, ready]
  );

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

export function useAuth() {
  const ctx = useContext(AuthContext);
  if (!ctx) throw new Error('useAuth must be used within AuthProvider');
  return ctx;
}
