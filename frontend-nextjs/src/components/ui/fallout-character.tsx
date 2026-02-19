'use client';

/**
 * Fallout / Vault Boy style character icon — simple cartoon figure with different poses.
 * Use next to section titles for a retro pip-boy feel.
 */
type Pose = 'storm' | 'arms' | 'resilience' | 'helm' | 'welcome' | 'how';

interface FalloutCharacterProps {
  pose: Pose;
  className?: string;
  size?: number;
}

const STROKE = 'currentColor';
const FILL = 'currentColor';

export function FalloutCharacter({ pose, className = '', size = 32 }: FalloutCharacterProps) {
  const s = size;
  const viewBox = '0 0 32 32';

  // Shared: head (circle)
  const head = <circle cx="16" cy="8" r="5" fill="none" stroke={STROKE} strokeWidth="1.5" />;
  // Body
  const body = <path d="M 16 14 L 16 22 L 14 24 L 18 24 Z" fill="none" stroke={STROKE} strokeWidth="1.2" />;

  const poses: Record<Pose, React.ReactNode> = {
    storm: (
      <>
        {head}
        {body}
        {/* Arms up / bracing */}
        <path d="M 10 12 L 14 16 M 22 12 L 18 16" stroke={STROKE} strokeWidth="1.2" fill="none" strokeLinecap="round" />
        <circle cx="16" cy="8" r="1.5" fill={FILL} />
      </>
    ),
    arms: (
      <>
        {head}
        {body}
        {/* Many arms out */}
        <path d="M 8 14 L 12 18 M 12 12 L 14 16 M 16 11 L 16 15 M 20 12 L 18 16 M 24 14 L 20 18" stroke={STROKE} strokeWidth="1.1" fill="none" strokeLinecap="round" />
        <circle cx="16" cy="8" r="1.5" fill={FILL} />
      </>
    ),
    resilience: (
      <>
        {head}
        {body}
        {/* Arms crossed */}
        <path d="M 10 16 L 18 20 M 22 16 L 14 20" stroke={STROKE} strokeWidth="1.2" fill="none" strokeLinecap="round" />
        <circle cx="16" cy="8" r="1.5" fill={FILL} />
      </>
    ),
    helm: (
      <>
        {head}
        {body}
        {/* One hand salute / pointing up */}
        <path d="M 10 15 L 12 12 L 12 10 M 22 16 L 20 18" stroke={STROKE} strokeWidth="1.2" fill="none" strokeLinecap="round" />
        <circle cx="16" cy="8" r="1.5" fill={FILL} />
      </>
    ),
    welcome: (
      <>
        {head}
        {body}
        {/* Thumbs up / wave */}
        <path d="M 10 14 L 12 16 L 10 18 M 22 14 L 20 18" stroke={STROKE} strokeWidth="1.2" fill="none" strokeLinecap="round" />
        <circle cx="16" cy="8" r="1.5" fill={FILL} />
      </>
    ),
    how: (
      <>
        {head}
        {body}
        {/* Thinking: hand to chin */}
        <path d="M 20 10 L 18 14 L 16 12 M 10 16 L 14 20" stroke={STROKE} strokeWidth="1.2" fill="none" strokeLinecap="round" />
        <circle cx="16" cy="8" r="1.5" fill={FILL} />
      </>
    ),
  };

  return (
    <svg
      viewBox={viewBox}
      width={s}
      height={s}
      className={`flex-shrink-0 text-amber-700 dark:text-amber-300 ${className}`}
      aria-hidden
    >
      {poses[pose]}
    </svg>
  );
}
