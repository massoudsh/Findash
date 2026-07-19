'use client';

import { useEffect, useState, useCallback } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Textarea } from '@/components/ui/textarea';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription,
  DialogFooter,
  DialogTrigger,
} from '@/components/ui/dialog';
import { toast } from '@/components/ui/toast';
import {
  Target,
  MessageSquare,
  TrendingUp,
  Plus,
  Trash2,
  Loader2,
  Users,
  ThumbsUp,
} from 'lucide-react';
import {
  getGTMHypotheses,
  createGTMHypothesis,
  deleteGTMHypothesis,
  getCustomerConversations,
  createCustomerConversation,
  deleteCustomerConversation,
  getTractionMetrics,
  createTractionMetric,
  deleteTractionMetric,
  getStartupTrackerSummary,
} from '@/lib/services/api';

interface GTMHypothesis {
  id: number;
  title: string;
  statement: string;
  target_segment: string;
  channel: string;
  status: 'untested' | 'testing' | 'validated' | 'invalidated';
  confidence: number;
  owner?: string | null;
  notes?: string | null;
  created_at: string;
}

interface CustomerConversation {
  id: number;
  contact_name: string;
  company?: string | null;
  role?: string | null;
  channel: string;
  sentiment: 'positive' | 'neutral' | 'negative';
  summary: string;
  key_insights: string[];
  hypothesis_id?: number | null;
  conversation_date: string;
}

interface TractionMetric {
  id: number;
  metric_name: string;
  category: 'users' | 'revenue' | 'engagement' | 'retention' | 'other';
  value: number;
  unit?: string | null;
  source?: string | null;
  notes?: string | null;
  recorded_at: string;
}

interface Summary {
  hypotheses_total: number;
  hypotheses_validated: number;
  hypotheses_testing: number;
  conversations_total: number;
  conversations_positive: number;
  traction_entries_total: number;
}

const statusLabel: Record<string, string> = {
  untested: 'آزمایش‌نشده',
  testing: 'در حال آزمایش',
  validated: 'تأیید‌شده',
  invalidated: 'رد‌شده',
};

const statusColor: Record<string, string> = {
  untested: 'bg-gray-100 text-gray-800',
  testing: 'bg-yellow-100 text-yellow-800',
  validated: 'bg-green-100 text-green-800',
  invalidated: 'bg-red-100 text-red-800',
};

const sentimentLabel: Record<string, string> = {
  positive: 'مثبت',
  neutral: 'خنثی',
  negative: 'منفی',
};

const sentimentColor: Record<string, string> = {
  positive: 'bg-green-100 text-green-800',
  neutral: 'bg-gray-100 text-gray-800',
  negative: 'bg-red-100 text-red-800',
};

const categoryLabel: Record<string, string> = {
  users: 'کاربران',
  revenue: 'درآمد',
  engagement: 'تعامل',
  retention: 'نگه‌داشت',
  other: 'سایر',
};

export function StartupTrackerPanel() {
  const [tab, setTab] = useState('hypotheses');
  const [loading, setLoading] = useState(true);
  const [summary, setSummary] = useState<Summary | null>(null);
  const [hypotheses, setHypotheses] = useState<GTMHypothesis[]>([]);
  const [conversations, setConversations] = useState<CustomerConversation[]>([]);
  const [traction, setTraction] = useState<TractionMetric[]>([]);

  const loadAll = useCallback(async () => {
    setLoading(true);
    try {
      const [sumRes, hypRes, convRes, tracRes] = await Promise.all([
        getStartupTrackerSummary(),
        getGTMHypotheses(),
        getCustomerConversations(),
        getTractionMetrics(),
      ]);
      setSummary(sumRes.data);
      setHypotheses(hypRes.data);
      setConversations(convRes.data);
      setTraction(tracRes.data);
    } catch (error) {
      toast({
        title: 'خطا در بارگذاری',
        description: 'اتصال به سرویس Startup Tracker ممکن نشد.',
        type: 'error',
      });
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    loadAll();
  }, [loadAll]);

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-2xl font-bold tracking-tight flex items-center gap-2">
          <Target className="w-6 h-6 text-purple-600" />
          استارتاپ‌تراکر (اعتبارسنجی محصول)
        </h2>
        <p className="text-muted-foreground text-sm">
          فرضیه‌های GTM، مکالمات با مشتری، و داده traction تیم Findash — فقط برای استفاده داخلی.
        </p>
      </div>

      <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
        <Card><CardContent className="p-4 text-center">
          <div className="text-2xl font-bold text-blue-600">{summary?.hypotheses_total ?? '—'}</div>
          <div className="text-xs text-gray-500">کل فرضیه‌ها</div>
        </CardContent></Card>
        <Card><CardContent className="p-4 text-center">
          <div className="text-2xl font-bold text-green-600">{summary?.hypotheses_validated ?? '—'}</div>
          <div className="text-xs text-gray-500">تأیید‌شده</div>
        </CardContent></Card>
        <Card><CardContent className="p-4 text-center">
          <div className="text-2xl font-bold text-yellow-600">{summary?.hypotheses_testing ?? '—'}</div>
          <div className="text-xs text-gray-500">در حال آزمایش</div>
        </CardContent></Card>
        <Card><CardContent className="p-4 text-center">
          <div className="text-2xl font-bold text-purple-600">{summary?.conversations_total ?? '—'}</div>
          <div className="text-xs text-gray-500">مکالمات ثبت‌شده</div>
        </CardContent></Card>
        <Card><CardContent className="p-4 text-center">
          <div className="text-2xl font-bold text-cyan-600">{summary?.traction_entries_total ?? '—'}</div>
          <div className="text-xs text-gray-500">داده traction</div>
        </CardContent></Card>
      </div>

      <Tabs value={tab} onValueChange={setTab} className="space-y-4">
        <TabsList className="grid w-full max-w-xl grid-cols-3">
          <TabsTrigger value="hypotheses" className="flex items-center gap-2">
            <Target className="w-4 h-4" /> فرضیه GTM
          </TabsTrigger>
          <TabsTrigger value="conversations" className="flex items-center gap-2">
            <MessageSquare className="w-4 h-4" /> مکالمه با مشتری
          </TabsTrigger>
          <TabsTrigger value="traction" className="flex items-center gap-2">
            <TrendingUp className="w-4 h-4" /> داده Traction
          </TabsTrigger>
        </TabsList>

        <TabsContent value="hypotheses">
          <HypothesesTab
            items={hypotheses}
            loading={loading}
            onChanged={loadAll}
          />
        </TabsContent>

        <TabsContent value="conversations">
          <ConversationsTab
            items={conversations}
            hypotheses={hypotheses}
            loading={loading}
            onChanged={loadAll}
          />
        </TabsContent>

        <TabsContent value="traction">
          <TractionTab
            items={traction}
            loading={loading}
            onChanged={loadAll}
          />
        </TabsContent>
      </Tabs>
    </div>
  );
}

// ---------------------------------------------------------------------------
// GTM Hypotheses tab
// ---------------------------------------------------------------------------

function HypothesesTab({
  items,
  loading,
  onChanged,
}: {
  items: GTMHypothesis[];
  loading: boolean;
  onChanged: () => void;
}) {
  const [open, setOpen] = useState(false);
  const [submitting, setSubmitting] = useState(false);
  const [form, setForm] = useState({
    title: '',
    statement: '',
    target_segment: '',
    channel: '',
    status: 'untested',
    confidence: 50,
    owner: '',
    notes: '',
  });

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    setSubmitting(true);
    try {
      await createGTMHypothesis({
        ...form,
        confidence: Number(form.confidence),
        owner: form.owner || null,
        notes: form.notes || null,
      });
      toast({ title: 'فرضیه ثبت شد', description: form.title, type: 'success' });
      setForm({ title: '', statement: '', target_segment: '', channel: '', status: 'untested', confidence: 50, owner: '', notes: '' });
      setOpen(false);
      onChanged();
    } catch {
      toast({ title: 'خطا', description: 'ثبت فرضیه ناموفق بود.', type: 'error' });
    } finally {
      setSubmitting(false);
    }
  }

  async function handleDelete(id: number) {
    try {
      await deleteGTMHypothesis(id);
      toast({ title: 'حذف شد', type: 'success' });
      onChanged();
    } catch {
      toast({ title: 'خطا', description: 'حذف فرضیه ناموفق بود.', type: 'error' });
    }
  }

  return (
    <Card>
      <CardHeader className="flex flex-row items-center justify-between">
        <CardTitle className="flex items-center gap-2">
          <Target className="w-5 h-5" /> فرضیه‌های GTM
        </CardTitle>
        <Dialog open={open} onOpenChange={setOpen}>
          <DialogTrigger asChild>
            <Button size="sm" className="flex items-center gap-2">
              <Plus className="w-4 h-4" /> فرضیه جدید
            </Button>
          </DialogTrigger>
          <DialogContent className="max-w-lg">
            <DialogHeader>
              <DialogTitle>ثبت فرضیه GTM جدید</DialogTitle>
              <DialogDescription>
                مثال: «ما باور داریم که [بخش بازار] به دلیل [علت] از [کانال] استفاده خواهند کرد.»
              </DialogDescription>
            </DialogHeader>
            <form onSubmit={handleSubmit} className="space-y-3">
              <div className="space-y-1">
                <Label>عنوان</Label>
                <Input value={form.title} onChange={(e) => setForm({ ...form, title: e.target.value })} required />
              </div>
              <div className="space-y-1">
                <Label>متن فرضیه</Label>
                <Textarea value={form.statement} onChange={(e) => setForm({ ...form, statement: e.target.value })} required />
              </div>
              <div className="grid grid-cols-2 gap-3">
                <div className="space-y-1">
                  <Label>بخش بازار هدف</Label>
                  <Input value={form.target_segment} onChange={(e) => setForm({ ...form, target_segment: e.target.value })} required />
                </div>
                <div className="space-y-1">
                  <Label>کانال</Label>
                  <Input value={form.channel} onChange={(e) => setForm({ ...form, channel: e.target.value })} required />
                </div>
              </div>
              <div className="grid grid-cols-2 gap-3">
                <div className="space-y-1">
                  <Label>وضعیت</Label>
                  <Select value={form.status} onValueChange={(v) => setForm({ ...form, status: v })}>
                    <SelectTrigger><SelectValue /></SelectTrigger>
                    <SelectContent>
                      {Object.entries(statusLabel).map(([k, v]) => (
                        <SelectItem key={k} value={k}>{v}</SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>
                <div className="space-y-1">
                  <Label>سطح اطمینان (٪)</Label>
                  <Input type="number" min={0} max={100} value={form.confidence}
                    onChange={(e) => setForm({ ...form, confidence: Number(e.target.value) })} />
                </div>
              </div>
              <div className="space-y-1">
                <Label>مسئول (اختیاری)</Label>
                <Input value={form.owner} onChange={(e) => setForm({ ...form, owner: e.target.value })} />
              </div>
              <div className="space-y-1">
                <Label>یادداشت (اختیاری)</Label>
                <Textarea value={form.notes} onChange={(e) => setForm({ ...form, notes: e.target.value })} />
              </div>
              <DialogFooter>
                <Button type="submit" disabled={submitting} className="flex items-center gap-2">
                  {submitting && <Loader2 className="w-4 h-4 animate-spin" />} ثبت فرضیه
                </Button>
              </DialogFooter>
            </form>
          </DialogContent>
        </Dialog>
      </CardHeader>
      <CardContent className="space-y-3">
        {loading && <div className="text-sm text-muted-foreground">در حال بارگذاری…</div>}
        {!loading && items.length === 0 && (
          <div className="text-sm text-muted-foreground">هنوز فرضیه‌ای ثبت نشده است.</div>
        )}
        {items.map((h) => (
          <div key={h.id} className="border rounded-lg p-4 space-y-2">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <span className="font-medium">{h.title}</span>
                <Badge className={statusColor[h.status]}>{statusLabel[h.status]}</Badge>
                <Badge variant="outline">اطمینان {h.confidence}٪</Badge>
              </div>
              <Button variant="outline" size="sm" className="text-red-600" onClick={() => handleDelete(h.id)}>
                <Trash2 className="w-4 h-4" />
              </Button>
            </div>
            <p className="text-sm text-muted-foreground">{h.statement}</p>
            <div className="flex flex-wrap gap-4 text-xs text-gray-500">
              <span>بخش هدف: {h.target_segment}</span>
              <span>کانال: {h.channel}</span>
              {h.owner && <span>مسئول: {h.owner}</span>}
            </div>
          </div>
        ))}
      </CardContent>
    </Card>
  );
}

// ---------------------------------------------------------------------------
// Customer Conversations tab
// ---------------------------------------------------------------------------

function ConversationsTab({
  items,
  hypotheses,
  loading,
  onChanged,
}: {
  items: CustomerConversation[];
  hypotheses: GTMHypothesis[];
  loading: boolean;
  onChanged: () => void;
}) {
  const [open, setOpen] = useState(false);
  const [submitting, setSubmitting] = useState(false);
  const [form, setForm] = useState({
    contact_name: '',
    company: '',
    role: '',
    channel: 'call',
    sentiment: 'neutral',
    summary: '',
    key_insights: '',
    hypothesis_id: '',
  });

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    setSubmitting(true);
    try {
      await createCustomerConversation({
        contact_name: form.contact_name,
        company: form.company || null,
        role: form.role || null,
        channel: form.channel,
        sentiment: form.sentiment,
        summary: form.summary,
        key_insights: form.key_insights
          .split('\n')
          .map((s) => s.trim())
          .filter(Boolean),
        hypothesis_id: form.hypothesis_id ? Number(form.hypothesis_id) : null,
      });
      toast({ title: 'مکالمه ثبت شد', description: form.contact_name, type: 'success' });
      setForm({ contact_name: '', company: '', role: '', channel: 'call', sentiment: 'neutral', summary: '', key_insights: '', hypothesis_id: '' });
      setOpen(false);
      onChanged();
    } catch {
      toast({ title: 'خطا', description: 'ثبت مکالمه ناموفق بود.', type: 'error' });
    } finally {
      setSubmitting(false);
    }
  }

  async function handleDelete(id: number) {
    try {
      await deleteCustomerConversation(id);
      toast({ title: 'حذف شد', type: 'success' });
      onChanged();
    } catch {
      toast({ title: 'خطا', description: 'حذف مکالمه ناموفق بود.', type: 'error' });
    }
  }

  return (
    <Card>
      <CardHeader className="flex flex-row items-center justify-between">
        <CardTitle className="flex items-center gap-2">
          <MessageSquare className="w-5 h-5" /> مکالمات با مشتری
        </CardTitle>
        <Dialog open={open} onOpenChange={setOpen}>
          <DialogTrigger asChild>
            <Button size="sm" className="flex items-center gap-2">
              <Plus className="w-4 h-4" /> ثبت مکالمه
            </Button>
          </DialogTrigger>
          <DialogContent className="max-w-lg">
            <DialogHeader>
              <DialogTitle>ثبت مکالمه جدید با مشتری</DialogTitle>
            </DialogHeader>
            <form onSubmit={handleSubmit} className="space-y-3">
              <div className="grid grid-cols-2 gap-3">
                <div className="space-y-1">
                  <Label>نام مخاطب</Label>
                  <Input value={form.contact_name} onChange={(e) => setForm({ ...form, contact_name: e.target.value })} required />
                </div>
                <div className="space-y-1">
                  <Label>شرکت (اختیاری)</Label>
                  <Input value={form.company} onChange={(e) => setForm({ ...form, company: e.target.value })} />
                </div>
              </div>
              <div className="grid grid-cols-2 gap-3">
                <div className="space-y-1">
                  <Label>کانال</Label>
                  <Select value={form.channel} onValueChange={(v) => setForm({ ...form, channel: v })}>
                    <SelectTrigger><SelectValue /></SelectTrigger>
                    <SelectContent>
                      <SelectItem value="call">تماس تلفنی</SelectItem>
                      <SelectItem value="meeting">جلسه حضوری</SelectItem>
                      <SelectItem value="email">ایمیل</SelectItem>
                      <SelectItem value="chat">چت</SelectItem>
                      <SelectItem value="survey">نظرسنجی</SelectItem>
                      <SelectItem value="other">سایر</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
                <div className="space-y-1">
                  <Label>احساس مشتری</Label>
                  <Select value={form.sentiment} onValueChange={(v) => setForm({ ...form, sentiment: v })}>
                    <SelectTrigger><SelectValue /></SelectTrigger>
                    <SelectContent>
                      {Object.entries(sentimentLabel).map(([k, v]) => (
                        <SelectItem key={k} value={k}>{v}</SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>
              </div>
              <div className="space-y-1">
                <Label>خلاصه مکالمه</Label>
                <Textarea value={form.summary} onChange={(e) => setForm({ ...form, summary: e.target.value })} required />
              </div>
              <div className="space-y-1">
                <Label>بینش‌های کلیدی (هر خط یک مورد)</Label>
                <Textarea value={form.key_insights} onChange={(e) => setForm({ ...form, key_insights: e.target.value })} />
              </div>
              <div className="space-y-1">
                <Label>فرضیه مرتبط (اختیاری)</Label>
                <Select value={form.hypothesis_id} onValueChange={(v) => setForm({ ...form, hypothesis_id: v })}>
                  <SelectTrigger><SelectValue placeholder="بدون ارتباط" /></SelectTrigger>
                  <SelectContent>
                    {hypotheses.map((h) => (
                      <SelectItem key={h.id} value={String(h.id)}>{h.title}</SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
              <DialogFooter>
                <Button type="submit" disabled={submitting} className="flex items-center gap-2">
                  {submitting && <Loader2 className="w-4 h-4 animate-spin" />} ثبت مکالمه
                </Button>
              </DialogFooter>
            </form>
          </DialogContent>
        </Dialog>
      </CardHeader>
      <CardContent className="space-y-3">
        {loading && <div className="text-sm text-muted-foreground">در حال بارگذاری…</div>}
        {!loading && items.length === 0 && (
          <div className="text-sm text-muted-foreground">هنوز مکالمه‌ای ثبت نشده است.</div>
        )}
        {items.map((c) => {
          const linkedHypothesis = hypotheses.find((h) => h.id === c.hypothesis_id);
          return (
            <div key={c.id} className="border rounded-lg p-4 space-y-2">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <span className="font-medium">{c.contact_name}</span>
                  {c.company && <span className="text-xs text-gray-500">({c.company})</span>}
                  <Badge className={sentimentColor[c.sentiment]}>{sentimentLabel[c.sentiment]}</Badge>
                </div>
                <Button variant="outline" size="sm" className="text-red-600" onClick={() => handleDelete(c.id)}>
                  <Trash2 className="w-4 h-4" />
                </Button>
              </div>
              <p className="text-sm text-muted-foreground">{c.summary}</p>
              {c.key_insights?.length > 0 && (
                <ul className="text-xs text-gray-600 list-disc pr-5 space-y-0.5">
                  {c.key_insights.map((k, i) => <li key={i}>{k}</li>)}
                </ul>
              )}
              {linkedHypothesis && (
                <Badge variant="outline" className="flex items-center gap-1 w-fit">
                  <Target className="w-3 h-3" /> مرتبط با: {linkedHypothesis.title}
                </Badge>
              )}
            </div>
          );
        })}
      </CardContent>
    </Card>
  );
}

// ---------------------------------------------------------------------------
// Traction Data tab
// ---------------------------------------------------------------------------

function TractionTab({
  items,
  loading,
  onChanged,
}: {
  items: TractionMetric[];
  loading: boolean;
  onChanged: () => void;
}) {
  const [open, setOpen] = useState(false);
  const [submitting, setSubmitting] = useState(false);
  const [form, setForm] = useState({
    metric_name: '',
    category: 'users',
    value: '',
    unit: '',
    source: '',
    notes: '',
  });

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    setSubmitting(true);
    try {
      await createTractionMetric({
        metric_name: form.metric_name,
        category: form.category,
        value: Number(form.value),
        unit: form.unit || null,
        source: form.source || null,
        notes: form.notes || null,
      });
      toast({ title: 'داده traction ثبت شد', description: form.metric_name, type: 'success' });
      setForm({ metric_name: '', category: 'users', value: '', unit: '', source: '', notes: '' });
      setOpen(false);
      onChanged();
    } catch {
      toast({ title: 'خطا', description: 'ثبت داده traction ناموفق بود.', type: 'error' });
    } finally {
      setSubmitting(false);
    }
  }

  async function handleDelete(id: number) {
    try {
      await deleteTractionMetric(id);
      toast({ title: 'حذف شد', type: 'success' });
      onChanged();
    } catch {
      toast({ title: 'خطا', description: 'حذف داده ناموفق بود.', type: 'error' });
    }
  }

  return (
    <Card>
      <CardHeader className="flex flex-row items-center justify-between">
        <CardTitle className="flex items-center gap-2">
          <TrendingUp className="w-5 h-5" /> داده‌های Traction
        </CardTitle>
        <Dialog open={open} onOpenChange={setOpen}>
          <DialogTrigger asChild>
            <Button size="sm" className="flex items-center gap-2">
              <Plus className="w-4 h-4" /> ثبت داده
            </Button>
          </DialogTrigger>
          <DialogContent className="max-w-lg">
            <DialogHeader>
              <DialogTitle>ثبت داده traction جدید</DialogTitle>
            </DialogHeader>
            <form onSubmit={handleSubmit} className="space-y-3">
              <div className="space-y-1">
                <Label>نام متریک</Label>
                <Input value={form.metric_name} onChange={(e) => setForm({ ...form, metric_name: e.target.value })}
                  placeholder="مثلاً کاربران فعال هفتگی" required />
              </div>
              <div className="grid grid-cols-2 gap-3">
                <div className="space-y-1">
                  <Label>دسته‌بندی</Label>
                  <Select value={form.category} onValueChange={(v) => setForm({ ...form, category: v })}>
                    <SelectTrigger><SelectValue /></SelectTrigger>
                    <SelectContent>
                      {Object.entries(categoryLabel).map(([k, v]) => (
                        <SelectItem key={k} value={k}>{v}</SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>
                <div className="space-y-1">
                  <Label>مقدار</Label>
                  <Input type="number" value={form.value} onChange={(e) => setForm({ ...form, value: e.target.value })} required />
                </div>
              </div>
              <div className="grid grid-cols-2 gap-3">
                <div className="space-y-1">
                  <Label>واحد (اختیاری)</Label>
                  <Input value={form.unit} onChange={(e) => setForm({ ...form, unit: e.target.value })} placeholder="کاربر، تومان، ٪..." />
                </div>
                <div className="space-y-1">
                  <Label>منبع (اختیاری)</Label>
                  <Input value={form.source} onChange={(e) => setForm({ ...form, source: e.target.value })} />
                </div>
              </div>
              <div className="space-y-1">
                <Label>یادداشت (اختیاری)</Label>
                <Textarea value={form.notes} onChange={(e) => setForm({ ...form, notes: e.target.value })} />
              </div>
              <DialogFooter>
                <Button type="submit" disabled={submitting} className="flex items-center gap-2">
                  {submitting && <Loader2 className="w-4 h-4 animate-spin" />} ثبت داده
                </Button>
              </DialogFooter>
            </form>
          </DialogContent>
        </Dialog>
      </CardHeader>
      <CardContent className="space-y-3">
        {loading && <div className="text-sm text-muted-foreground">در حال بارگذاری…</div>}
        {!loading && items.length === 0 && (
          <div className="text-sm text-muted-foreground">هنوز داده‌ای ثبت نشده است.</div>
        )}
        {items.map((t) => (
          <div key={t.id} className="border rounded-lg p-4 flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="w-9 h-9 rounded-full bg-cyan-100 flex items-center justify-center">
                {t.category === 'users' ? <Users className="w-4 h-4 text-cyan-600" /> :
                  t.category === 'revenue' ? <TrendingUp className="w-4 h-4 text-cyan-600" /> :
                    <ThumbsUp className="w-4 h-4 text-cyan-600" />}
              </div>
              <div>
                <div className="font-medium">{t.metric_name}</div>
                <div className="text-xs text-gray-500">
                  {categoryLabel[t.category]} {t.source && `• منبع: ${t.source}`}
                </div>
              </div>
            </div>
            <div className="flex items-center gap-3">
              <div className="text-lg font-bold">
                {t.value.toLocaleString('fa-IR')} {t.unit || ''}
              </div>
              <Button variant="outline" size="sm" className="text-red-600" onClick={() => handleDelete(t.id)}>
                <Trash2 className="w-4 h-4" />
              </Button>
            </div>
          </div>
        ))}
      </CardContent>
    </Card>
  );
}
