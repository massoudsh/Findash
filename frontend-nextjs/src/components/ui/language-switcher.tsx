'use client';

import { useLocale } from '@/lib/i18n/locale-context';
import { Button } from '@/components/ui/button';
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu';
import { Languages } from 'lucide-react';

export function LanguageSwitcher() {
  const { locale, setLocale, t } = useLocale();

  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>
        <Button variant="ghost" size="icon" className="h-9 w-9" aria-label={t('common.language')}>
          <Languages className="h-4 w-4" />
        </Button>
      </DropdownMenuTrigger>
      <DropdownMenuContent align="end">
        <DropdownMenuItem onClick={() => setLocale('en')}>
          {t('common.english')}
          {locale === 'en' && ' ✓'}
        </DropdownMenuItem>
        <DropdownMenuItem onClick={() => setLocale('es')}>
          {t('common.spanish')}
          {locale === 'es' && ' ✓'}
        </DropdownMenuItem>
        <DropdownMenuItem onClick={() => setLocale('fa')}>
          {t('common.persian')}
          {locale === 'fa' && ' ✓'}
        </DropdownMenuItem>
      </DropdownMenuContent>
    </DropdownMenu>
  );
}
