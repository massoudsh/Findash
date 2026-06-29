import { NextRequest, NextResponse } from 'next/server';

export interface NewsItem {
  id: string;
  title: string;
  description: string;
  url: string;
  source: string;
  category: 'gold' | 'currency' | 'stock' | 'crypto' | 'macro' | 'general';
  publishedAt: string;
  lang: 'fa' | 'en';
}

/** Parse a simple RSS/Atom feed (plain text) into NewsItem[]. */
function parseRSS(xml: string, source: string, lang: 'fa' | 'en', category: NewsItem['category']): NewsItem[] {
  const items: NewsItem[] = [];
  const itemRegex = /<item[^>]*>([\s\S]*?)<\/item>/gi;
  let match: RegExpExecArray | null;
  let idx = 0;
  while ((match = itemRegex.exec(xml)) !== null && items.length < 10) {
    const block = match[1];
    const get = (tag: string) => {
      const m = block.match(new RegExp(`<${tag}[^>]*><!\\[CDATA\\[([\\s\\S]*?)\\]\\]><\\/${tag}>|<${tag}[^>]*>([\\s\\S]*?)<\\/${tag}>`));
      return m ? (m[1] ?? m[2] ?? '').trim() : '';
    };
    const title = get('title');
    const link = get('link');
    const desc = get('description');
    const pubDate = get('pubDate') || get('dc:date') || new Date().toISOString();
    if (!title) continue;
    items.push({
      id: `${source}-${idx++}`,
      title,
      description: desc.replace(/<[^>]+>/g, '').slice(0, 200),
      url: link,
      source,
      category,
      publishedAt: pubDate,
      lang,
    });
  }
  return items;
}

const FEEDS = [
  { url: 'https://www.tgju.org/rss/news', source: 'TGJU', lang: 'fa' as const, category: 'gold' as const },
  { url: 'https://www.tgju.org/rss/currency', source: 'TGJU Currency', lang: 'fa' as const, category: 'currency' as const },
  { url: 'https://www.eghtesadonline.com/rss/', source: 'Eghtesad Online', lang: 'fa' as const, category: 'macro' as const },
];

// Fallback mock news when feeds are unavailable
function mockNews(): NewsItem[] {
  const now = new Date();
  return [
    { id: 'm1', title: 'قیمت طلا امروز به ۴ میلیون تومان رسید', description: 'قیمت هر گرم طلای ۱۸ عیار امروز با رشد قابل توجهی روبرو شد.', url: 'https://tgju.org', source: 'TGJU', category: 'gold', publishedAt: new Date(now.getTime() - 1 * 3600000).toISOString(), lang: 'fa' },
    { id: 'm2', title: 'نرخ دلار در بازار آزاد به ۶۵۰۰۰ تومان رسید', description: 'دلار آمریکا در بازار آزاد تهران امروز با نرخ ۶۵,۰۰۰ تومان معامله شد.', url: 'https://tgju.org', source: 'TGJU Currency', category: 'currency', publishedAt: new Date(now.getTime() - 2 * 3600000).toISOString(), lang: 'fa' },
    { id: 'm3', title: 'شاخص بورس اوراق بهادار تهران ۵۰۰۰ واحد رشد کرد', description: 'شاخص کل بورس تهران در پایان معاملات امروز با رشد ۵۰۰۰ واحدی به سطح ۲.۱ میلیون واحد رسید.', url: 'https://tgju.org', source: 'Eghtesad Online', category: 'stock', publishedAt: new Date(now.getTime() - 3 * 3600000).toISOString(), lang: 'fa' },
    { id: 'm4', title: 'Bitcoin tops $70,000 as Iran crypto demand rises', description: 'Bitcoin surpassed $70,000 amid increased demand from Iranian traders seeking dollar alternatives.', url: 'https://tgju.org', source: 'Crypto News', category: 'crypto', publishedAt: new Date(now.getTime() - 4 * 3600000).toISOString(), lang: 'en' },
    { id: 'm5', title: 'نرخ تورم ایران در آبان ماه اعلام شد', description: 'مرکز آمار ایران نرخ تورم ماه آبان را ۴۰.۵ درصد اعلام کرد.', url: 'https://tgju.org', source: 'Eghtesad Online', category: 'macro', publishedAt: new Date(now.getTime() - 5 * 3600000).toISOString(), lang: 'fa' },
    { id: 'm6', title: 'قیمت سکه بهار آزادی امروز', description: 'سکه بهار آزادی طرح جدید امروز با قیمت ۳۸ میلیون تومان معامله شد.', url: 'https://tgju.org', source: 'TGJU', category: 'gold', publishedAt: new Date(now.getTime() - 6 * 3600000).toISOString(), lang: 'fa' },
    { id: 'm7', title: 'Iran oil exports reach 6-month high', description: 'Iranian oil exports climbed to the highest level in six months, easing pressure on the rial.', url: 'https://tgju.org', source: 'Energy News', category: 'macro', publishedAt: new Date(now.getTime() - 7 * 3600000).toISOString(), lang: 'en' },
    { id: 'm8', title: 'نوسانات ارزی و تاثیر آن بر بازار سهام', description: 'کارشناسان معتقدند نوسانات اخیر نرخ ارز تاثیر مستقیمی بر سودآوری شرکت‌های صادراتی دارد.', url: 'https://tgju.org', source: 'Eghtesad Online', category: 'general', publishedAt: new Date(now.getTime() - 8 * 3600000).toISOString(), lang: 'fa' },
  ];
}

export async function GET(request: NextRequest) {
  const { searchParams } = new URL(request.url);
  const category = searchParams.get('category') || 'all';

  const allNews: NewsItem[] = [];

  // Try fetching real RSS feeds with short timeout
  const results = await Promise.allSettled(
    FEEDS.map(async (feed) => {
      const res = await fetch(feed.url, {
        signal: AbortSignal.timeout(4000),
        headers: { 'Accept': 'application/rss+xml, application/xml, text/xml' },
      });
      if (!res.ok) throw new Error(`${feed.source}: ${res.status}`);
      const text = await res.text();
      return parseRSS(text, feed.source, feed.lang, feed.category);
    })
  );

  results.forEach((r) => {
    if (r.status === 'fulfilled') allNews.push(...r.value);
  });

  // Use mocks if real feeds failed
  const news = allNews.length > 0 ? allNews : mockNews();

  const filtered = category === 'all' ? news : news.filter((n) => n.category === category);

  return NextResponse.json({
    status: allNews.length > 0 ? 'live' : 'mock',
    count: filtered.length,
    data: filtered.sort((a, b) => new Date(b.publishedAt).getTime() - new Date(a.publishedAt).getTime()),
  }, {
    headers: { 'Cache-Control': 'public, s-maxage=120, stale-while-revalidate=60' },
  });
}
