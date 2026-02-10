import React from 'react';

interface OctopusMascotProps {
  className?: string;
  size?: 'sm' | 'md' | 'lg';
}

export function OctopusMascot({ className = '', size = 'md' }: OctopusMascotProps) {
  const sizeClasses = {
    sm: 'w-16 h-16',
    md: 'w-24 h-24',
    lg: 'w-32 h-32'
  };

  return (
    <div className={`relative ${sizeClasses[size]} ${className}`}>
      {/* Octopus body with subtle animation */}
      <div className="relative w-full h-full">
        {/* Main body */}
        <div className="absolute inset-0 bg-gradient-to-b from-blue-400 to-blue-600 rounded-full shadow-lg animate-float">
          {/* Eyes */}
          <div className="absolute top-3 left-1/2 transform -translate-x-1/2 flex gap-2">
            <div className="w-3 h-3 bg-white rounded-full relative">
              <div className="absolute top-0.5 left-0.5 w-2 h-2 bg-slate-800 rounded-full animate-pulse"></div>
            </div>
            <div className="w-3 h-3 bg-white rounded-full relative">
              <div className="absolute top-0.5 left-0.5 w-2 h-2 bg-slate-800 rounded-full animate-pulse"></div>
            </div>
          </div>
          
          {/* Smile */}
          <div className="absolute top-8 left-1/2 transform -translate-x-1/2 w-4 h-2 border-b-2 border-white rounded-full"></div>
        </div>
        
        {/* Tentacles */}
        <div className="absolute -bottom-2 left-1/2 transform -translate-x-1/2 flex gap-1">
          {[...Array(4)].map((_, i) => (
            <div
              key={i}
              className="w-2 h-6 bg-gradient-to-b from-blue-500 to-blue-700 rounded-full"
              style={{
                animation: `float 2s ease-in-out infinite`,
                animationDelay: `${i * 0.2}s`
              }}
            ></div>
          ))}
        </div>
      </div>
    </div>
  );
} 