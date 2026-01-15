import type { CapacitorConfig } from '@capacitor/cli';

const config: CapacitorConfig = {
  appId: 'com.localbooru.app',
  appName: 'LocalBooru',
  webDir: 'dist',
  server: {
    // In development, connect to local server
    // In production, the app serves its own files
    androidScheme: 'https',
  },
  plugins: {
    Preferences: {
      // For storing server list
    },
  },
  android: {
    allowMixedContent: true, // Allow HTTP connections to local servers
  },
  ios: {
    // iOS-specific config
  },
};

export default config;
