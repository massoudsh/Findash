'use client';

import { useState, useEffect } from 'react';
import { Monitor, Moon, Sun } from 'lucide-react';
import { Button } from '@/components/ui/button';

type Theme = 'light' | 'dark' | 'system';

interface ThemeSwitcherProps {
  className?: string;
  showLabel?: boolean;
}

export function ThemeSwitcher({ className, showLabel = false }: ThemeSwitcherProps) {
  const [theme, setTheme] = useState<Theme>('system');
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    setMounted(true);
    const stored = localStorage.getItem('theme') as Theme;
    if (stored) {
      setTheme(stored);
    }
  }, []);

  useEffect(() => {
    if (!mounted) return;

    const root = window.document.documentElement;
    
    if (theme === 'system') {
      const systemTheme = window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
      root.classList.remove('light', 'dark');
      root.classList.add(systemTheme);
    } else {
      root.classList.remove('light', 'dark');
      root.classList.add(theme);
    }

    localStorage.setItem('theme', theme);
  }, [theme, mounted]);

  useEffect(() => {
    if (!mounted) return;

    const mediaQuery = window.matchMedia('(prefers-color-scheme: dark)');
    const handleChange = () => {
      if (theme === 'system') {
        const root = window.document.documentElement;
        const systemTheme = mediaQuery.matches ? 'dark' : 'light';
        root.classList.remove('light', 'dark');
        root.classList.add(systemTheme);
      }
    };

    mediaQuery.addEventListener('change', handleChange);
    return () => mediaQuery.removeEventListener('change', handleChange);
  }, [theme, mounted]);

  const cycleTheme = () => {
    const themes: Theme[] = ['light', 'dark', 'system'];
    const currentIndex = themes.indexOf(theme);
    const nextTheme = themes[(currentIndex + 1) % themes.length];
    setTheme(nextTheme);
  };

  if (!mounted) {
    return (
      <Button variant="ghost" size="sm" disabled className={className}>
        <Sun className="h-4 w-4" />
        {showLabel && <span className="ml-2">پوسته</span>}
      </Button>
    );
  }

  const getIcon = () => {
    switch (theme) {
      case 'light':
        return <Sun className="h-4 w-4" />;
      case 'dark':
        return <Moon className="h-4 w-4" />;
      case 'system':
        return <Monitor className="h-4 w-4" />;
    }
  };

  const getLabel = () => {
    switch (theme) {
      case 'light':
        return 'روشن';
      case 'dark':
        return 'تیره';
      case 'system':
        return 'سیستم';
    }
  };

  return (
    <Button 
      variant="ghost" 
      size="sm" 
      onClick={cycleTheme}
      className={className}
      title={`پوسته فعلی: ${getLabel()}. برای تغییر کلیک کنید.`}
    >
      {getIcon()}
      {showLabel && <span className="ml-2">{getLabel()}</span>}
    </Button>
  );
} 