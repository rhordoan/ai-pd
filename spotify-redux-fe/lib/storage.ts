import {
  getItemAsync as secureGet,
  setItemAsync as secureSet,
  deleteItemAsync as secureDelete,
  isAvailableAsync as secureAvailable,
} from 'expo-secure-store';

async function canUseSecureStore() {
  try {
    return typeof secureAvailable === 'function' && (await secureAvailable());
  } catch {
    return false;
  }
}

export async function storageGet(key: string): Promise<string | null> {
  try {
    if (await canUseSecureStore()) return await secureGet(key);
  } catch {
    // fall through
  }
  try {
    if (typeof window !== 'undefined' && window.localStorage) return window.localStorage.getItem(key);
  } catch {
    // ignore
  }
  return null;
}

export async function storageSet(key: string, value: string): Promise<void> {
  try {
    if (await canUseSecureStore()) {
      await secureSet(key, value);
      return;
    }
  } catch {
    // fall through
  }
  try {
    if (typeof window !== 'undefined' && window.localStorage) window.localStorage.setItem(key, value);
  } catch {
    // ignore
  }
}

export async function storageDelete(key: string): Promise<void> {
  try {
    if (await canUseSecureStore()) {
      await secureDelete(key);
      return;
    }
  } catch {
    // fall through
  }
  try {
    if (typeof window !== 'undefined' && window.localStorage) window.localStorage.removeItem(key);
  } catch {
    // ignore
  }
}

