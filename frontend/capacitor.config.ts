import type { CapacitorConfig } from '@capacitor/cli';

const config: CapacitorConfig = {
  appId: 'com.localbooru.app',
  appName: 'LocalBooru',
  webDir: 'dist',
  server: {
    // Use https scheme for security (self-signed certs allowed via network config)
    androidScheme: 'https',
  },
  plugins: {
    Preferences: {
      // For storing server list
    },
  },
  android: {
    allowMixedContent: true, // Allow HTTP connections to legacy local servers
  },
  ios: {
    // iOS-specific config
  },
};

export default config;
