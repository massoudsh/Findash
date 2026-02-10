import * as React from "react"

import { cn } from "@/lib/utils"

interface CardProps extends React.HTMLAttributes<HTMLDivElement> {
  variant?: "default" | "glass" | "gradient" | "elevated" | "bordered";
  hover?: boolean;
}

const Card = React.forwardRef<HTMLDivElement, CardProps>(
  ({ className, variant = "default", hover = true, ...props }, ref) => {
    const variants = {
      default: "rounded-xl border border-border/50 bg-card/50 backdrop-blur-sm shadow-lg shadow-black/5 dark:shadow-black/20",
      glass: "rounded-xl border border-white/10 bg-white/5 backdrop-blur-xl shadow-xl shadow-black/10 dark:bg-black/20 dark:border-white/5",
      gradient: "rounded-xl border-0 bg-gradient-to-br from-card via-card to-card/80 backdrop-blur-sm shadow-xl shadow-primary/5",
      elevated: "rounded-2xl border border-border/40 bg-card shadow-2xl shadow-black/10 dark:shadow-black/30",
      bordered: "rounded-xl border-2 border-border bg-card/80 backdrop-blur-sm shadow-lg shadow-black/5",
    };

    const hoverClasses = hover
      ? "transition-all duration-300 hover:shadow-xl hover:shadow-black/10 hover:-translate-y-1 hover:border-primary/20 dark:hover:shadow-black/30"
      : "";

    return (
      <div
        ref={ref}
        className={cn(
          variants[variant],
          hoverClasses,
          className
        )}
        {...props}
      />
    );
  }
);
Card.displayName = "Card"

const CardHeader = React.forwardRef<
  HTMLDivElement,
  React.HTMLAttributes<HTMLDivElement>
>(({ className, ...props }, ref) => (
  <div
    ref={ref}
    className={cn(
      "flex flex-col space-y-2 p-6 pb-4",
      className
    )}
    {...props}
  />
))
CardHeader.displayName = "CardHeader"

const CardTitle = React.forwardRef<
  HTMLParagraphElement,
  React.HTMLAttributes<HTMLHeadingElement>
>(({ className, ...props }, ref) => (
  <h3
    ref={ref}
    className={cn(
      "text-2xl font-bold leading-tight tracking-tight bg-gradient-to-r from-foreground to-foreground/70 bg-clip-text text-transparent",
      className
    )}
    {...props}
  />
))
CardTitle.displayName = "CardTitle"

const CardDescription = React.forwardRef<
  HTMLParagraphElement,
  React.HTMLAttributes<HTMLParagraphElement>
>(({ className, ...props }, ref) => (
  <p
    ref={ref}
    className={cn(
      "text-sm font-medium text-muted-foreground leading-relaxed",
      className
    )}
    {...props}
  />
))
CardDescription.displayName = "CardDescription"

const CardContent = React.forwardRef<
  HTMLDivElement,
  React.HTMLAttributes<HTMLDivElement>
>(({ className, ...props }, ref) => (
  <div 
    ref={ref} 
    className={cn("p-6 pt-0", className)} 
    {...props} 
  />
))
CardContent.displayName = "CardContent"

const CardFooter = React.forwardRef<
  HTMLDivElement,
  React.HTMLAttributes<HTMLDivElement>
>(({ className, ...props }, ref) => (
  <div
    ref={ref}
    className={cn(
      "flex items-center p-6 pt-4 border-t border-border/50",
      className
    )}
    {...props}
  />
))
CardFooter.displayName = "CardFooter"

// Modern card variants
const GlassCard = React.forwardRef<HTMLDivElement, React.HTMLAttributes<HTMLDivElement>>(
  ({ className, ...props }, ref) => (
    <Card
      ref={ref}
      variant="glass"
      className={cn("group", className)}
      {...props}
    />
  )
);
GlassCard.displayName = "GlassCard"

const GradientCard = React.forwardRef<HTMLDivElement, React.HTMLAttributes<HTMLDivElement>>(
  ({ className, ...props }, ref) => (
    <Card
      ref={ref}
      variant="gradient"
      className={cn("group relative overflow-hidden", className)}
      {...props}
    />
  )
);
GradientCard.displayName = "GradientCard"

const ElevatedCard = React.forwardRef<HTMLDivElement, React.HTMLAttributes<HTMLDivElement>>(
  ({ className, ...props }, ref) => (
    <Card
      ref={ref}
      variant="elevated"
      className={cn("group", className)}
      {...props}
    />
  )
);
ElevatedCard.displayName = "ElevatedCard"

export { 
  Card, 
  CardHeader, 
  CardFooter, 
  CardTitle, 
  CardDescription, 
  CardContent,
  GlassCard,
  GradientCard,
  ElevatedCard
} 