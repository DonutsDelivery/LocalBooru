import type { CapacitorConfig } from '@capacitor/cli';

const config: CapacitorConfig = {
  appId: 'com.localbooru.app',
  appName: 'LocalBooru',
  webDir: 'dist',
  server: {
    // Use http scheme to allow loading resources from local http servers
    androidScheme: 'http',
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
