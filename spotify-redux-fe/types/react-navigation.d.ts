// Ambient React Navigation namespace expected by downstream packages.
// Exposes the Theme type so TypeScript can resolve ReactNavigation.Theme.
declare global {
  namespace ReactNavigation {
    type Theme = import('@react-navigation/native').Theme;
  }
}

export {};
