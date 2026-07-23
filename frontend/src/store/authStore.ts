import { create } from 'zustand';

interface AuthStore {
  token: string | null;
  user: { username: string; role: string } | null;
  setAuth: (token: string, role: string, username: string) => void;
  logout: () => void;
}

const getInitialToken = () => {
  return localStorage.getItem('token') || null;
};

const getInitialUser = () => {
  const role = localStorage.getItem('role');
  const username = localStorage.getItem('username');
  if (role && username) {
    return { role, username };
  }
  return null;
};

export const useAuthStore = create<AuthStore>((set) => ({
  token: getInitialToken(),
  user: getInitialUser(),
  setAuth: (token, role, username) => {
    localStorage.setItem('token', token);
    localStorage.setItem('role', role);
    localStorage.setItem('username', username);
    set({ token, user: { role, username } });
  },
  logout: () => {
    localStorage.removeItem('token');
    localStorage.removeItem('role');
    localStorage.removeItem('username');
    set({ token: null, user: null });
  },
}));
