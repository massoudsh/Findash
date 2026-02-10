import React from 'react';
import Image from 'next/image';

interface OctopusLogoProps {
  size?: number;
  className?: string;
  showText?: boolean;
  variant?: 'svg' | 'image' | 'text-only';
  textSize?: 'sm' | 'md' | 'lg' | 'xl';
}

export function OctopusLogo({ 
  size = 32, 
  className = "", 
  showText = true, 
  variant = 'svg',
  textSize = 'md' 
}: OctopusLogoProps) {
  const textSizeClasses = {
    sm: 'text-sm',
    md: 'text-lg',
    lg: 'text-xl',
    xl: 'text-3xl font-bold'
  };

  const LogoSVG = () => (
    <svg 
      xmlns="http://www.w3.org/2000/svg" 
      viewBox="0 0 32 32" 
      width={size} 
      height={size}
      className="flex-shrink-0"
    >
             {/* Gradient definitions for bright high-contrast octopus colors */}
       <defs>
         <radialGradient id="octopusGradient" cx="50%" cy="30%" r="70%">
           <stop offset="0%" style={{ stopColor: "#10b981", stopOpacity: 1 }} />
           <stop offset="100%" style={{ stopColor: "#059669", stopOpacity: 1 }} />
         </radialGradient>
         <linearGradient id="tentacleGradient" x1="0%" y1="0%" x2="100%" y2="100%">
           <stop offset="0%" style={{ stopColor: "#34d399", stopOpacity: 1 }} />
           <stop offset="100%" style={{ stopColor: "#10b981", stopOpacity: 1 }} />
         </linearGradient>
       </defs>

      {/* Main octopus body */}
      <circle cx="16" cy="14" r="10" fill="url(#octopusGradient)" />
      
      {/* Eyes - large and friendly */}
      <circle cx="12" cy="12" r="2.5" fill="white" />
      <circle cx="20" cy="12" r="2.5" fill="white" />
      <circle cx="12" cy="12" r="1.6" fill="#1e293b"/>
      <circle cx="20" cy="12" r="1.6" fill="#1e293b"/>
      
      {/* Eye sparkles */}
      <circle cx="12.8" cy="11.2" r="0.5" fill="white"/>
      <circle cx="20.8" cy="11.2" r="0.5" fill="white"/>
      
      {/* Happy smile */}
      <path d="M 11 16 Q 16 18.5 21 16" stroke="#1e293b" strokeWidth="1.2" fill="none" strokeLinecap="round"/>
      
      {/* Stylized tentacles */}
      <path d="M 8 22 Q 6 26 8 28 Q 9 29 10 28" stroke="url(#tentacleGradient)" strokeWidth="2.5" fill="none" strokeLinecap="round"/>
      <path d="M 12 24 Q 10 28 12 30 Q 13 31 14 30" stroke="url(#tentacleGradient)" strokeWidth="2.5" fill="none" strokeLinecap="round"/>
      <path d="M 16 24 Q 16 28 16 30 Q 16 31 16 30" stroke="url(#tentacleGradient)" strokeWidth="2.5" fill="none" strokeLinecap="round"/>
      <path d="M 20 24 Q 22 28 20 30 Q 19 31 18 30" stroke="url(#tentacleGradient)" strokeWidth="2.5" fill="none" strokeLinecap="round"/>
      <path d="M 24 22 Q 26 26 24 28 Q 23 29 22 28" stroke="url(#tentacleGradient)" strokeWidth="2.5" fill="none" strokeLinecap="round"/>
      
             {/* Cute cheek spots */}
       <ellipse cx="8" cy="13" rx="1.5" ry="1" fill="#6ee7b7" opacity="0.8"/>
       <ellipse cx="24" cy="13" rx="1.5" ry="1" fill="#6ee7b7" opacity="0.8"/>
    </svg>
  );

  const LogoImage = () => (
    <Image
      src="/octopus-logo-with-text.jpg"
      alt="Octopus Trading Platform"
      width={size}
      height={size}
      className="rounded-lg object-contain"
    />
  );

  const TextOnlyImage = () => (
    <Image
      src="/octopus-text-only.jpg"
      alt="Octopus"
      width={size * 3}
      height={size}
      className="object-contain"
    />
  );

  if (variant === 'text-only') {
    return (
      <div className={`inline-flex items-center ${className}`}>
        <TextOnlyImage />
      </div>
    );
  }

  if (variant === 'image') {
    return (
      <div className={`inline-flex items-center space-x-3 ${className}`}>
        <LogoImage />
        {showText && (
          <div className="flex flex-col">
            <span className={`font-bold text-foreground ${textSizeClasses[textSize]}`}>
              Octopus
            </span>
            <span className="text-xs text-muted-foreground">
              Trading Platform
            </span>
          </div>
        )}
      </div>
    );
  }

  return (
    <div className={`inline-flex items-center space-x-3 ${className}`}>
      <LogoSVG />
      {showText && (
        <div className="flex flex-col">
          <span className={`font-bold text-foreground ${textSizeClasses[textSize]}`}>
            Octopus
          </span>
          <span className="text-xs text-muted-foreground">
            Trading Platform
          </span>
        </div>
      )}
    </div>
  );
} 