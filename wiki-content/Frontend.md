# Frontend Architecture

The Octopus Trading Platform frontend is built with Next.js 14, TypeScript, and Tailwind CSS, featuring a modern glassmorphism design.

## Technology Stack

| Technology | Version | Purpose |
|------------|---------|---------|
| Next.js | 14.x | React framework |
| TypeScript | 5.x | Type-safe JavaScript |
| Tailwind CSS | 3.x | Utility-first styling |
| Shadcn UI | Latest | Component library |
| Radix UI | Latest | Headless components |
| Recharts | 2.x | Data visualization |

---

## Project Structure

```
frontend-nextjs/
├── src/
│   ├── app/                    # Next.js App Router
│   │   ├── layout.tsx          # Root layout
│   │   ├── page.tsx            # Home page
│   │   ├── dashboard/          # Dashboard pages
│   │   ├── trading/            # Trading pages
│   │   ├── portfolios/         # Portfolio pages
│   │   ├── market-data/        # Market data pages
│   │   ├── agents/             # AI agents pages
│   │   ├── risk/               # Risk management
│   │   └── settings/           # Settings pages
│   ├── components/
│   │   ├── ui/                 # Base UI components
│   │   │   ├── button.tsx
│   │   │   ├── card.tsx
│   │   │   ├── input.tsx
│   │   │   └── ...
│   │   ├── dashboard/          # Dashboard components
│   │   ├── navigation/         # Navigation components
│   │   └── charts/             # Chart components
│   ├── lib/
│   │   ├── utils.ts            # Utility functions
│   │   └── api.ts              # API client
│   └── styles/
│       └── globals.css         # Global styles
├── public/                     # Static assets
├── tailwind.config.ts          # Tailwind configuration
├── next.config.js              # Next.js configuration
└── package.json
```

---

## Design System

### Color Palette

```css
/* CSS Variables */
:root {
  --background: 222.2 84% 4.9%;
  --foreground: 210 40% 98%;
  --card: 222.2 84% 4.9%;
  --card-foreground: 210 40% 98%;
  --primary: 217.2 91.2% 59.8%;
  --primary-foreground: 222.2 47.4% 11.2%;
  --secondary: 217.2 32.6% 17.5%;
  --accent: 217.2 32.6% 17.5%;
  --muted: 217.2 32.6% 17.5%;
  --border: 217.2 32.6% 17.5%;
}
```

### Typography

```typescript
// Font configuration
const fontSans = {
  family: 'Inter, system-ui, sans-serif',
  weights: [400, 500, 600, 700]
};
```

---

## Component Library

### Card Components

The platform uses modern glassmorphism card styles with multiple variants:

```typescript
// Card variants
const cardVariants = {
  default: "bg-card/50 backdrop-blur-sm border-border/50",
  glass: "bg-white/5 backdrop-blur-xl border-white/10",
  gradient: "bg-gradient-to-br from-primary/20 to-secondary/20",
  elevated: "shadow-xl shadow-black/10",
  bordered: "border-2 border-primary/20"
};
```

**Usage:**
```tsx
import { Card, GlassCard, ElevatedCard } from "@/components/ui/card";

// Default card
<Card>Content</Card>

// Glass effect card
<GlassCard>Content</GlassCard>

// Elevated card with shadow
<ElevatedCard>Content</ElevatedCard>
```

### Navigation

Dual-sidebar navigation with grouped items:

```typescript
// Left sidebar - Main features
const leftSidebarItems = {
  overview: [
    { name: "Dashboard", href: "/", icon: Home },
    { name: "Market Data", href: "/market-data", icon: LineChart }
  ],
  trading: [
    { name: "Trading", href: "/trading", icon: TrendingUp },
    { name: "Portfolios", href: "/portfolios", icon: Briefcase }
  ]
};

// Right sidebar - Tools & settings
const rightSidebarItems = {
  analytics: [
    { name: "Backtesting", href: "/backtesting", icon: History },
    { name: "Analytics", href: "/analytics", icon: BarChart3 }
  ],
  system: [
    { name: "Settings", href: "/settings", icon: Settings }
  ]
};
```

---

## Page Examples

### Dashboard Page

```tsx
// app/dashboard/page.tsx
import { Suspense } from "react";
import { DashboardContent } from "@/components/dashboard/dashboard-content";
import { GlassCard } from "@/components/ui/card";

export default function DashboardPage() {
  return (
    <div className="space-y-6 p-6">
      {/* Market Status Bar */}
      <GlassCard className="p-4">
        <MarketStatusBar />
      </GlassCard>
      
      {/* Main Dashboard Content */}
      <Suspense fallback={<DashboardSkeleton />}>
        <DashboardContent />
      </Suspense>
    </div>
  );
}
```

### API Integration

```typescript
// lib/api.ts
const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

export async function fetchMarketData(symbol: string) {
  const response = await fetch(`${API_BASE_URL}/api/market/quote/${symbol}`, {
    headers: {
      'Authorization': `Bearer ${getToken()}`,
      'Content-Type': 'application/json'
    }
  });
  
  if (!response.ok) {
    throw new Error('Failed to fetch market data');
  }
  
  return response.json();
}
```

---

## State Management

### Client-side State

Using React hooks for local state:

```typescript
// Example: Portfolio state
const [portfolios, setPortfolios] = useState<Portfolio[]>([]);
const [isLoading, setIsLoading] = useState(true);
const [error, setError] = useState<string | null>(null);

useEffect(() => {
  async function loadPortfolios() {
    try {
      const data = await fetchPortfolios();
      setPortfolios(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setIsLoading(false);
    }
  }
  loadPortfolios();
}, []);
```

### URL State with nuqs

```typescript
// Using nuqs for URL search params
import { useQueryState } from 'nuqs';

function SymbolFilter() {
  const [symbol, setSymbol] = useQueryState('symbol');
  
  return (
    <Input 
      value={symbol || ''} 
      onChange={(e) => setSymbol(e.target.value)}
      placeholder="Search symbol..."
    />
  );
}
```

---

## Styling Guidelines

### Tailwind Classes

```tsx
// Responsive design
<div className="
  grid 
  grid-cols-1 
  md:grid-cols-2 
  lg:grid-cols-3 
  gap-4
">

// Glassmorphism effect
<div className="
  bg-white/5 
  backdrop-blur-xl 
  border 
  border-white/10 
  rounded-xl
">

// Gradient backgrounds
<div className="
  bg-gradient-to-br 
  from-blue-500/20 
  to-purple-500/20
">

// Hover effects
<button className="
  transition-all 
  duration-300 
  hover:shadow-xl 
  hover:-translate-y-1
">
```

### Animation Classes

```tsx
// Pulse animation for live indicators
<span className="animate-pulse bg-green-500 rounded-full h-2 w-2" />

// Fade in animation
<div className="animate-in fade-in duration-500">

// Slide up animation
<div className="animate-in slide-in-from-bottom duration-300">
```

---

## Performance Optimization

### Server Components

```tsx
// Prefer Server Components (default)
async function MarketDataPage() {
  const data = await fetchMarketData(); // Server-side fetch
  return <MarketDataTable data={data} />;
}

// Use 'use client' only when necessary
'use client';
function InteractiveChart({ data }) {
  // Client-side interactivity
}
```

### Dynamic Imports

```tsx
import dynamic from 'next/dynamic';

// Lazy load heavy components
const TradingChart = dynamic(
  () => import('@/components/charts/trading-chart'),
  { 
    loading: () => <ChartSkeleton />,
    ssr: false 
  }
);
```

### Image Optimization

```tsx
import Image from 'next/image';

<Image
  src="/logo.png"
  alt="Logo"
  width={200}
  height={50}
  priority // For above-the-fold images
/>
```

---

## WebSocket Integration

```typescript
// Real-time data connection
import { useEffect, useState } from 'react';

function useMarketData(symbols: string[]) {
  const [data, setData] = useState<MarketData[]>([]);
  
  useEffect(() => {
    const ws = new WebSocket('ws://localhost:8000/ws/market-data');
    
    ws.onopen = () => {
      ws.send(JSON.stringify({
        type: 'subscribe',
        symbols
      }));
    };
    
    ws.onmessage = (event) => {
      const update = JSON.parse(event.data);
      setData(prev => updateMarketData(prev, update));
    };
    
    return () => ws.close();
  }, [symbols]);
  
  return data;
}
```

---

## Development Commands

```bash
# Start development server
npm run dev

# Build for production
npm run build

# Start production server
npm start

# Run linter
npm run lint

# Type check
npm run type-check
```

---

## Environment Variables

```bash
# .env.local
NEXT_PUBLIC_API_URL=http://localhost:8000
NEXT_PUBLIC_WS_URL=ws://localhost:8000
NEXT_PUBLIC_APP_NAME=Octopus Trading
```

---

## Next Steps

- [[Architecture]] - Overall system architecture
- [[API Reference]] - Backend API documentation
- [[Configuration]] - Environment configuration
