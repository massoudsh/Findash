'use client';

import { FormEvent, useCallback, useEffect, useState } from 'react';
import { AlertCircle, CalendarDays, ClipboardList, Eye, Landmark, Plus, Search, WalletCards } from 'lucide-react';
import { Alert } from '@/components/ui/alert';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';

const toman = new Intl.NumberFormat('fa-IR');
const formatToman = (value: number) => `${toman.format(value)} تومان`;

type MarketItem = { symbol: string; label: string; category: string; price: number; change_pct: number };
type Watchlist = { id: string; name: string; symbols: string[] };
type Dividend = { id: string; symbol: string; amount: number; currency: string; paid_at: string; status: 'received' | 'expected'; note: string };

export default function InvestingPage() {
  const [market, setMarket] = useState<MarketItem[]>([]);
  const [watchlists, setWatchlists] = useState<Watchlist[]>([]);
  const [dividends, setDividends] = useState<Dividend[]>([]);
  const [cash, setCash] = useState<number | null>(null);
  const [query, setQuery] = useState('');
  const [category, setCategory] = useState('');
  const [watchlistName, setWatchlistName] = useState('');
  const [symbols, setSymbols] = useState('');
  const [message, setMessage] = useState<string | null>(null);

  const load = useCallback(async () => {
    try {
      const [screen, lists, paper, dividendData] = await Promise.all([
        fetch('/api/investor-tools/screener').then(r => r.json()),
        fetch('/api/investor-tools/watchlists').then(r => r.json()),
        fetch('/api/investor-tools/paper').then(r => r.json()),
        fetch('/api/investor-tools/dividends').then(r => r.json()),
      ]);
      setMarket(screen.items ?? []);
      setWatchlists(lists ?? []);
      setCash(paper.cash ?? null);
      setDividends(dividendData.items ?? []);
    } catch {
      setMessage('دریافت داده‌ها ممکن نشد. اتصال را بررسی و دوباره تلاش کنید.');
    }
  }, []);

  useEffect(() => { load(); }, [load]);

  async function createWatchlist(event: FormEvent) {
    event.preventDefault();
    const response = await fetch('/api/investor-tools/watchlists', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name: watchlistName, symbols: symbols.split(',').map(s => s.trim()).filter(Boolean) }),
    });
    if (!response.ok) return setMessage((await response.json()).detail ?? 'ساخت واچ‌لیست ناموفق بود.');
    setWatchlistName(''); setSymbols(''); setMessage('واچ‌لیست ذخیره شد.'); load();
  }

  async function manageWatchlist(list: Watchlist, action: 'edit' | 'delete') {
    if (action === 'delete') {
      if (!window.confirm(`واچ‌لیست «${list.name}» حذف شود؟`)) return;
      const response = await fetch(`/api/investor-tools/watchlists/${list.id}`, { method: 'DELETE' });
      if (!response.ok) return setMessage('حذف واچ‌لیست ناموفق بود.');
      setMessage('واچ‌لیست حذف شد.'); load();
      return;
    }
    const name = window.prompt('نام واچ‌لیست', list.name);
    if (!name?.trim()) return;
    const nextSymbols = window.prompt('نمادها را با ویرگول جدا کنید', list.symbols.join(', '));
    if (nextSymbols === null) return;
    const response = await fetch(`/api/investor-tools/watchlists/${list.id}`, {
      method: 'PUT', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name, symbols: nextSymbols.split(',').map(symbol => symbol.trim()).filter(Boolean) }),
    });
    if (!response.ok) return setMessage('ویرایش واچ‌لیست ناموفق بود.');
    setMessage('واچ‌لیست به‌روزرسانی شد.'); load();
  }

  async function placePaperOrder(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const data = new FormData(event.currentTarget);
    const response = await fetch('/api/investor-tools/paper/orders', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ symbol: data.get('symbol'), side: data.get('side'), quantity: Number(data.get('quantity')), price: Number(data.get('price')), thesis: data.get('thesis') }),
    });
    if (!response.ok) return setMessage((await response.json()).detail ?? 'ثبت سفارش ناموفق بود.');
    setMessage('سفارش فقط در حساب آزمایشی ثبت شد.'); (event.target as HTMLFormElement).reset(); load();
  }

  async function addDividend(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const data = new FormData(event.currentTarget);
    const response = await fetch('/api/investor-tools/dividends', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ symbol: data.get('symbol'), amount: Number(data.get('amount')), paid_at: data.get('paid_at'), status: data.get('status'), currency: 'IRT', note: data.get('note') }),
    });
    if (!response.ok) return setMessage('ثبت سود نقدی ناموفق بود.');
    setMessage('سود نقدی ثبت شد.'); (event.target as HTMLFormElement).reset(); load();
  }

  const screened = market.filter(item =>
    (!category || item.category === category) &&
    (`${item.symbol} ${item.label}`.toLowerCase().includes(query.toLowerCase()))
  );

  return <main className="space-y-6" dir="rtl">
    <header>
      <h1 className="text-3xl font-bold tracking-tight">ابزار سرمایه‌گذاری</h1>
      <p className="text-muted-foreground">رصد بازار، شبیه‌سازی معامله و ثبت بازده پرتفوی در یک فضای فارسی.</p>
    </header>
    {message && <Alert><AlertCircle className="h-4 w-4" /><div className="flex-1">{message}</div><Button variant="ghost" size="sm" onClick={() => setMessage(null)}>بستن</Button></Alert>}

    <Tabs defaultValue="watchlists" className="space-y-5">
      <TabsList className="grid h-auto w-full grid-cols-2 gap-1 md:grid-cols-5">
        <TabsTrigger value="watchlists"><Eye className="ml-2 h-4 w-4" />واچ‌لیست</TabsTrigger>
        <TabsTrigger value="screener"><Search className="ml-2 h-4 w-4" />اسکرینر</TabsTrigger>
        <TabsTrigger value="paper"><WalletCards className="ml-2 h-4 w-4" />حساب آزمایشی</TabsTrigger>
        <TabsTrigger value="calendar"><CalendarDays className="ml-2 h-4 w-4" />تقویم مالی</TabsTrigger>
        <TabsTrigger value="dividends"><Landmark className="ml-2 h-4 w-4" />سود نقدی</TabsTrigger>
      </TabsList>

      <TabsContent value="watchlists" className="grid gap-5 lg:grid-cols-[360px_1fr]">
        <Card><CardHeader><CardTitle>واچ‌لیست جدید</CardTitle></CardHeader><CardContent>
          <form onSubmit={createWatchlist} className="space-y-4">
            <div><Label htmlFor="watchlist-name">نام</Label><Input id="watchlist-name" value={watchlistName} onChange={e => setWatchlistName(e.target.value)} required /></div>
            <div><Label htmlFor="watchlist-symbols">نمادها</Label><Input id="watchlist-symbols" value={symbols} onChange={e => setSymbols(e.target.value)} placeholder="BTC-IRT, GOLD18-IRT" /></div>
            <p className="text-xs text-muted-foreground">حداکثر ۵ واچ‌لیست؛ نمادها را با ویرگول جدا کنید.</p>
            <Button className="w-full" type="submit"><Plus className="ml-2 h-4 w-4" />ذخیره واچ‌لیست</Button>
          </form>
        </CardContent></Card>
        <section className="grid gap-4 md:grid-cols-2">{watchlists.length ? watchlists.map(list => <Card key={list.id}><CardHeader className="flex flex-row items-center justify-between"><CardTitle className="text-lg">{list.name}</CardTitle><div className="flex gap-1"><Button variant="ghost" size="sm" onClick={() => manageWatchlist(list, 'edit')}>ویرایش</Button><Button variant="ghost" size="sm" className="text-destructive" onClick={() => manageWatchlist(list, 'delete')}>حذف</Button></div></CardHeader><CardContent className="flex flex-wrap gap-2">{list.symbols.length ? list.symbols.map(symbol => <Badge key={symbol} variant="secondary" dir="ltr">{symbol}</Badge>) : <span className="text-sm text-muted-foreground">هنوز نمادی ندارد.</span>}</CardContent></Card>) : <Card className="md:col-span-2"><CardContent className="py-10 text-center text-muted-foreground">اولین واچ‌لیست خود را بسازید.</CardContent></Card>}</section>
      </TabsContent>

      <TabsContent value="screener" className="space-y-4"><Card><CardHeader><CardTitle>اسکرینر بازار ایران</CardTitle></CardHeader><CardContent className="space-y-4">
        <div className="grid gap-3 md:grid-cols-2"><Input value={query} onChange={e => setQuery(e.target.value)} placeholder="جست‌وجوی نام یا نماد" /><select aria-label="دسته دارایی" value={category} onChange={e => setCategory(e.target.value)} className="h-10 rounded-md border bg-background px-3"><option value="">همه دسته‌ها</option><option value="currency">ارز</option><option value="gold">طلا</option><option value="coin">سکه</option><option value="crypto">رمزارز</option></select></div>
        <div className="overflow-x-auto"><table className="w-full text-sm"><thead className="border-b text-right text-muted-foreground"><tr><th className="p-3">دارایی</th><th className="p-3">قیمت</th><th className="p-3">تغییر امروز</th><th className="p-3">دسته</th></tr></thead><tbody>{screened.map(item => <tr key={item.symbol} className="border-b"><td className="p-3"><strong dir="ltr">{item.symbol}</strong><span className="mr-2 text-muted-foreground">{item.label}</span></td><td className="p-3">{formatToman(item.price)}</td><td className={`p-3 ${item.change_pct >= 0 ? 'text-emerald-600' : 'text-red-600'}`}>{toman.format(item.change_pct)}٪</td><td className="p-3"><Badge variant="outline">{item.category}</Badge></td></tr>)}</tbody></table></div>
      </CardContent></Card></TabsContent>

      <TabsContent value="paper" className="grid gap-5 lg:grid-cols-[360px_1fr]"><Card><CardHeader><CardTitle>ثبت سفارش آزمایشی</CardTitle></CardHeader><CardContent><form onSubmit={placePaperOrder} className="space-y-4"><div><Label htmlFor="paper-symbol">نماد</Label><Input id="paper-symbol" name="symbol" defaultValue="BTC-IRT" required /></div><div className="grid grid-cols-2 gap-3"><div><Label htmlFor="paper-side">نوع</Label><select id="paper-side" name="side" className="h-10 w-full rounded-md border bg-background px-3"><option value="buy">خرید</option><option value="sell">فروش</option></select></div><div><Label htmlFor="paper-quantity">تعداد</Label><Input id="paper-quantity" name="quantity" type="number" min="0.0001" step="any" required /></div></div><div><Label htmlFor="paper-price">قیمت (تومان)</Label><Input id="paper-price" name="price" type="number" min="1" required /></div><div><Label htmlFor="paper-thesis">منطق معامله</Label><Input id="paper-thesis" name="thesis" /></div><Button className="w-full" type="submit">ثبت در حساب آزمایشی</Button></form></CardContent></Card><Card><CardHeader><CardTitle>وضعیت حساب</CardTitle></CardHeader><CardContent className="space-y-3"><p className="text-3xl font-bold">{cash === null ? '—' : formatToman(cash)}</p><p className="text-muted-foreground">موجودی قابل استفاده</p><Alert><AlertCircle className="h-4 w-4" />تمام سفارش‌های این بخش شبیه‌سازی‌شده‌اند و به هیچ کارگزار یا سرویس پرداختی ارسال نمی‌شوند.</Alert></CardContent></Card></TabsContent>

      <TabsContent value="calendar"><Card><CardHeader><CardTitle>تقویم مالی و رویدادهای بازار</CardTitle></CardHeader><CardContent className="space-y-3"><Alert><CalendarDays className="h-4 w-4" />رویدادهای رسمی پس از اتصال منبع تأییدشده نمایش داده می‌شوند؛ هیچ دادهٔ نمونه‌ای به‌عنوان رویداد واقعی نمایش داده نمی‌شود.</Alert><p className="text-sm text-muted-foreground">این ماژول آمادهٔ دریافت مجمع، سود نقدی، افزایش سرمایه، عرضه اولیه و رویدادهای کلان با تاریخ شمسی است.</p></CardContent></Card></TabsContent>

      <TabsContent value="dividends" className="grid gap-5 lg:grid-cols-[360px_1fr]"><Card><CardHeader><CardTitle>ثبت سود نقدی</CardTitle></CardHeader><CardContent><form onSubmit={addDividend} className="space-y-4"><div><Label htmlFor="dividend-symbol">نماد</Label><Input id="dividend-symbol" name="symbol" required /></div><div><Label htmlFor="dividend-amount">مبلغ (تومان)</Label><Input id="dividend-amount" name="amount" type="number" min="1" required /></div><div><Label htmlFor="dividend-date">تاریخ پرداخت</Label><Input id="dividend-date" name="paid_at" type="date" required /></div><select name="status" aria-label="وضعیت سود" className="h-10 w-full rounded-md border bg-background px-3"><option value="received">دریافت‌شده</option><option value="expected">در انتظار</option></select><Input name="note" placeholder="یادداشت اختیاری" /><Button className="w-full" type="submit"><ClipboardList className="ml-2 h-4 w-4" />ثبت سود نقدی</Button></form></CardContent></Card><Card><CardHeader><CardTitle>تاریخچه سود نقدی</CardTitle></CardHeader><CardContent className="space-y-3">{dividends.length ? dividends.map(item => <div key={item.id} className="flex items-center justify-between border-b pb-3"><div><strong dir="ltr">{item.symbol}</strong><p className="text-xs text-muted-foreground">{item.paid_at}</p></div><div className="text-left"><b>{formatToman(item.amount)}</b><p className="text-xs text-muted-foreground">{item.status === 'received' ? 'دریافت‌شده' : 'در انتظار'}</p></div></div>) : <p className="py-8 text-center text-muted-foreground">سود نقدی ثبت نشده است.</p>}</CardContent></Card></TabsContent>
    </Tabs>
  </main>;
}
