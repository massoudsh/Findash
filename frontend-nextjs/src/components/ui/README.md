# UI Components

## OctopusLogo Component

The `OctopusLogo` component provides a flexible way to display the Octopus Trading Platform branding throughout the application.

### Props

- `size?: number` - Size of the logo in pixels (default: 32)
- `className?: string` - Additional CSS classes
- `showText?: boolean` - Whether to show the "Octopus Trading Platform" text (default: true)
- `variant?: 'svg' | 'image' | 'text-only'` - Logo variant to use (default: 'svg')
- `textSize?: 'sm' | 'md' | 'lg' | 'xl'` - Size of the text when shown (default: 'md')

### Variants

1. **svg** (default) - Uses the custom SVG octopus logo with modern gradient colors
2. **image** - Uses the `/octopus-logo-with-text.jpg` image file
3. **text-only** - Uses the `/octopus-text-only.jpg` image file

### Usage Examples

```tsx
// Default usage with SVG logo and text
<OctopusLogo />

// Large logo without text (for mobile navigation)
<OctopusLogo size={32} showText={false} />

// Desktop sidebar with text
<OctopusLogo size={36} showText={true} textSize="lg" />

// Using image variant
<OctopusLogo variant="image" size={40} />

// Text-only variant
<OctopusLogo variant="text-only" />
```

### Design Features

- Modern indigo/purple gradient colors matching the platform's design system
- Responsive and scalable SVG implementation
- Professional typography with "Octopus" as the main brand name
- "Trading Platform" as descriptive subtitle
- Optimized for both light and dark themes

### Assets

The component can use the following image assets from `/public`:
- `octopus-logo-with-text.jpg` - Full logo with text
- `octopus-text-only.jpg` - Text-only branding
- `favicon.svg` - Favicon version of the logo 