/** @type {import("next").NextConfig} */
const nextConfig = {
  // Performance optimizations
  reactStrictMode: true,
  compress: true,

  // Command Center: Options and Bots
  async redirects() {
    return [
      { source: '/realtime', destination: '/trading', permanent: false },
      { source: '/trades', destination: '/trading', permanent: false },
      { source: '/trading-bots', destination: '/trading?tab=bots', permanent: false },
      { source: '/strategies', destination: '/trading?tab=strategies', permanent: false },
      { source: '/backtesting', destination: '/trading?tab=strategies&subtab=backtesting', permanent: false },
      { source: '/risk', destination: '/trading?tab=risk', permanent: false },
      { source: '/data-explorer', destination: '/data', permanent: false },
      { source: '/visualization', destination: '/data?tab=charts', permanent: false },
      { source: '/profile', destination: '/account', permanent: false },
      { source: '/settings', destination: '/account?tab=settings', permanent: false },
      { source: '/audit-log', destination: '/admin?tab=audit', permanent: false },
    ];
  },
  
  // Output configuration
  output: 'standalone',
  
  // Image optimization
  images: {
    domains: ["localhost"],
    formats: ['image/avif', 'image/webp'],
  },
  
  // Experimental features for speed
  experimental: {
    optimizePackageImports: ['lucide-react', '@radix-ui/react-icons'],
  },
  
  // On-demand entries for faster dev server
  onDemandEntries: {
    maxInactiveAge: 60 * 1000,
    pagesBufferLength: 5,
  },
}

module.exports = nextConfig
