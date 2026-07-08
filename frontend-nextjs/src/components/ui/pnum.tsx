/**
 * PNum — نمایش اعداد به فارسی
 * استفاده: <PNum>{value}</PNum> یا <PNum value={1234} />
 */

import { toPersian } from '@/lib/fa-utils';

interface PNumProps {
  value?: number | string;
  children?: number | string;
  className?: string;
}

export function PNum({ value, children, className }: PNumProps) {
  const raw = value ?? children ?? '';
  return (
    <span className={className}>{toPersian(String(raw))}</span>
  );
}
