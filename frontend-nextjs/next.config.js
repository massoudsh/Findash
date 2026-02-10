/** @type {import("next").NextConfig} */
const nextConfig = {
  // Performance optimizations
  reactStrictMode: true,
  compress: true,
  
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
