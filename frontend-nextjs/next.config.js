/** @type {import("next").NextConfig} */
const nextConfig = {
  // Performance optimizations
  reactStrictMode: true,
  compress: true,

  // Trading Center: Market, Live Trading, and Bots in one page
  async redirects() {
    return [
      { source: '/realtime', destination: '/trading?tab=market', permanent: false },
      { source: '/trades', destination: '/trading?tab=center', permanent: false },
      { source: '/trading-bots', destination: '/trading?tab=bots', permanent: false },
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
