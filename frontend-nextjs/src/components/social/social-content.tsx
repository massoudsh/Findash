'use client';

import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Progress } from '@/components/ui/progress';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { formatDate } from '@/lib/utils';
import {
  MessageSquare,
  TrendingUp,
  TrendingDown,
  Twitter,
  Globe,
  Users,
  Activity,
  Eye,
  Heart,
  MessageCircle,
  Share2,
  ExternalLink,
  Bot,
  Crown,
  Zap,
  AlertTriangle,
  CheckCircle,
  Clock,
  Hash,
  Flame,
  Target,
  RefreshCw
} from 'lucide-react';

interface SentimentData {
  symbol: string;
  sentiment: 'bullish' | 'bearish' | 'neutral';
  score: number;
  mentions: number;
  change24h: number;
  volume: number;
  influence: number;
  platforms: {
    twitter: number;
    discord: number;
    lunar: number;
    reddit: number;
  };
}

interface SocialPost {
  id: string;
  platform: 'twitter' | 'discord' | 'lunar' | 'reddit' | 'telegram';
  author: string;
  content: string;
  sentiment: 'positive' | 'negative' | 'neutral';
  mentions: string[];
  timestamp: string;
  engagement: {
    likes: number;
    retweets: number;
    comments: number;
    views: number;
  };
  verified: boolean;
  influence_score: number;
  channel?: string;
  tags: string[];
}

interface InfluencerData {
  id: string;
  username: string;
  platform: string;
  followers: number;
  verified: boolean;
  influence_score: number;
  recent_sentiment: 'bullish' | 'bearish' | 'neutral';
  accuracy_rate: number;
  specialties: string[];
  avatar: string;
}

interface TrendingTopic {
  hashtag: string;
  mentions: number;
  sentiment: number;
  change24h: number;
  platforms: string[];
  related_assets: string[];
}

interface DiscordChannel {
  name: string;
  server: string;
  members: number;
  activity_score: number;
  sentiment: 'bullish' | 'bearish' | 'neutral';
  recent_messages: number;
  top_discussed: string[];
  verified: boolean;
}

export function SocialContent() {
  const [socialData, setSocialData] = useState<any>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [lastUpdated, setLastUpdated] = useState<string>('');

  // Fetch real social data
  const fetchSocialData = async () => {
    setIsLoading(true);
    try {
      const response = await fetch('/api/social/comprehensive');
      if (response.ok) {
        const data = await response.json();
        setSocialData(data.data);
        setLastUpdated(data.last_updated);
      } else {
        console.error('Failed to fetch social data');
        // Fall back to mock data
        setSocialData(getMockSocialData());
      }
    } catch (error) {
      console.error('Error fetching social data:', error);
      // Fall back to mock data
      setSocialData(getMockSocialData());
    } finally {
      setIsLoading(false);
    }
  };

  // Mock data fallback
  const getMockSocialData = () => ({
    fear_greed_index: {
      current_value: 68,
      sentiment: 'greed',
      sentiment_text: 'Greed',
      change_24h: 3
    },
    twitter_sentiment: {
      overall_sentiment: { score: 68.5, sentiment: 'bullish', volume: 145230 },
      top_coins: {
        BTC: { mentions: 28470, sentiment_score: 72.3, sentiment: 'bullish' },
        ETH: { mentions: 19230, sentiment_score: 68.7, sentiment: 'bullish' }
      }
    },
    reddit_sentiment: {
      Bitcoin: { sentiment_score: 75.0, mentions: 25, sentiment: 'bullish' },
      CryptoCurrency: { sentiment_score: 65.0, mentions: 20, sentiment: 'slightly_bullish' }
    },
    volume_trends: {
      platforms: {
        twitter: { volume_24h: 145230, change_24h: 12.5 },
        reddit: { volume_24h: 8940, change_24h: -3.1 }
      }
    }
  });

  useEffect(() => {
    fetchSocialData();
    // Auto-refresh every 10 minutes
    const interval = setInterval(fetchSocialData, 10 * 60 * 1000);
    return () => clearInterval(interval);
  }, []);

  // Convert real data to expected format
  const sentimentData: SentimentData[] = socialData ? [
    {
      symbol: 'BTC',
      sentiment: socialData.twitter_sentiment?.top_coins?.BTC?.sentiment || 'bullish',
      score: socialData.twitter_sentiment?.top_coins?.BTC?.sentiment_score || 78.5,
      mentions: socialData.twitter_sentiment?.top_coins?.BTC?.mentions || 28470,
      change24h: 12.3,
      volume: 1250000,
      influence: 85.2,
      platforms: { twitter: 15420, discord: 8930, lunar: 2850, reddit: 1270 },
    },
    {
      symbol: 'ETH',
      sentiment: 'bullish',
      score: 71.2,
      mentions: 19230,
      change24h: 8.7,
      volume: 890000,
      influence: 79.8,
      platforms: { twitter: 12100, discord: 4200, lunar: 1980, reddit: 950 },
    },
    {
      symbol: 'SOL',
      sentiment: 'neutral',
      score: 52.1,
      mentions: 16540,
      change24h: -2.1,
      volume: 450000,
      influence: 68.4,
      platforms: { twitter: 9800, discord: 3900, lunar: 1840, reddit: 1000 },
    },
    {
      symbol: 'DOGE',
      sentiment: 'bearish',
      score: 34.8,
      mentions: 45210,
      change24h: -15.6,
      volume: 2100000,
      influence: 45.2,
      platforms: { twitter: 32000, discord: 8900, lunar: 3200, reddit: 1110 },
    },
    {
      symbol: 'PEPE',
      sentiment: 'bullish',
      score: 82.3,
      mentions: 32140,
      change24h: 18.9,
      volume: 1800000,
      influence: 72.1,
      platforms: { twitter: 18500, discord: 9800, lunar: 2940, reddit: 900 },
    },
  ] : [];

  const [socialPosts] = useState<SocialPost[]>([
    {
      id: '1',
      platform: 'lunar',
      author: 'CryptoWhaleAlert',
      content: '🚨 انباشت عظیم بیت‌کوین شناسایی شد 🚨\n\nسه کیف پول نهنگ به‌تازگی ۲٬۸۴۷ بیت‌کوین (۱۲۰ میلیون دلار) را به ذخیره‌سازی سرد منتقل کردند. پول هوشمند پیش از موج صعودی بعدی در حال جمع‌آوری است! 📈\n\n#بیت‌کوین #هشدار_نهنگ #روند_صعودی',
      sentiment: 'positive',
      mentions: ['BTC'],
      timestamp: '2024-01-20T10:30:00Z',
      engagement: { likes: 2847, retweets: 1203, comments: 456, views: 28470 },
      verified: true,
      influence_score: 92.5,
      tags: ['هشدار-نهنگ', 'انباشت', 'صعودی'],
      channel: 'whale-alerts'
    },
    {
      id: '2',
      platform: 'discord',
      author: 'DegenTrader#1337',
      content: 'تازه تو پوزیشن لانگ سولانا داغون شدم... بازارسازها دوباره دنبال استاپ‌ها هستن. این دستکاری بازار داره مسخره میشه. شاید برای معاملات امن‌تر به اتریوم روی بیارم.',
      sentiment: 'negative',
      mentions: ['SOL', 'ETH'],
      timestamp: '2024-01-20T09:15:00Z',
      engagement: { likes: 156, retweets: 45, comments: 89, views: 2340 },
      verified: false,
      influence_score: 34.2,
      tags: ['معامله', 'ضرر', 'دستکاری-بازار'],
      channel: 'Alpha Traders'
    },
    {
      id: '3',
      platform: 'twitter',
      author: '@ElonMusk',
      content: 'دوج‌کوین، کریپتوی مردم است. نه اوج، نه فرود، فقط داج. 🐕',
      sentiment: 'positive',
      mentions: ['DOGE'],
      timestamp: '2024-01-20T08:45:00Z',
      engagement: { likes: 45230, retweets: 18940, comments: 8970, views: 1200000 },
      verified: true,
      influence_score: 98.9,
      tags: ['داج', 'میم', 'کریپتوی-مردم']
    },
    {
      id: '4',
      platform: 'lunar',
      author: 'PepeMoonMission',
      content: '🐸 ارتش پپه جمع شوید! 🐸\n\nمشارکت‌های جدید اعلام شد:\n✅ لیست شدن در یک صرافی متمرکز بزرگ، هفته آینده\n✅ عرضه مجموعه NFT\n✅ فعال شدن پاداش استیکینگ\n\nاین فقط شروع ماجراست! 🚀🌙',
      sentiment: 'positive',
      mentions: ['PEPE'],
      timestamp: '2024-01-19T16:22:00Z',
      engagement: { likes: 3420, retweets: 1890, comments: 567, views: 34200 },
      verified: true,
      influence_score: 78.4,
      tags: ['پپه', 'مشارکت‌ها', 'صعودی'],
      channel: 'meme-coins'
    },
    {
      id: '5',
      platform: 'discord',
      author: 'VitalikButerin#0001',
      content: 'راهکارهای مقیاس‌پذیری اتریوم عملکرد فوق‌العاده‌ای دارند. پذیرش لایه دوم سریع‌تر از حد انتظار در حال شتاب گرفتن است. آینده امور مالی غیرمتمرکز روشن است.',
      sentiment: 'positive',
      mentions: ['ETH'],
      timestamp: '2024-01-19T14:10:00Z',
      engagement: { likes: 5670, retweets: 2340, comments: 890, views: 45600 },
      verified: true,
      influence_score: 95.8,
      tags: ['اتریوم', 'مقیاس‌پذیری', 'لایه-دو'],
      channel: 'Ethereum Foundation'
    }
  ]);

  const [influencers] = useState<InfluencerData[]>([
    {
      id: '1',
      username: '@ElonMusk',
      platform: 'Twitter',
      followers: 170000000,
      verified: true,
      influence_score: 98.9,
      recent_sentiment: 'bullish',
      accuracy_rate: 67.8,
      specialties: ['DOGE', 'BTC', 'فناوری'],
      avatar: '🚀'
    },
    {
      id: '2',
      username: 'CryptoWhaleAlert',
      platform: 'Lunar',
      followers: 2400000,
      verified: true,
      influence_score: 92.5,
      recent_sentiment: 'bullish',
      accuracy_rate: 84.2,
      specialties: ['BTC', 'ETH', 'ردیابی نهنگ‌ها'],
      avatar: '🐋'
    },
    {
      id: '3',
      username: 'VitalikButerin#0001',
      platform: 'Discord',
      followers: 890000,
      verified: true,
      influence_score: 95.8,
      recent_sentiment: 'bullish',
      accuracy_rate: 91.5,
      specialties: ['ETH', 'DeFi', 'L2'],
      avatar: 'Ⓥ'
    },
    {
      id: '4',
      username: 'PepeMoonMission',
      platform: 'Lunar',
      followers: 450000,
      verified: true,
      influence_score: 78.4,
      recent_sentiment: 'bullish',
      accuracy_rate: 72.3,
      specialties: ['PEPE', 'کوین‌های میم'],
      avatar: '🐸'
    }
  ]);

  const [trendingTopics] = useState<TrendingTopic[]>([
    {
      hashtag: '#BitcoinETF',
      mentions: 45200,
      sentiment: 78.5,
      change24h: 234.5,
      platforms: ['Twitter', 'Discord', 'Lunar'],
      related_assets: ['BTC']
    },
    {
      hashtag: '#PepeSeason',
      mentions: 32400,
      sentiment: 82.1,
      change24h: 567.8,
      platforms: ['Lunar', 'Discord', 'Twitter'],
      related_assets: ['PEPE']
    },
    {
      hashtag: '#EthereumUpgrade',
      mentions: 28900,
      sentiment: 71.2,
      change24h: 89.3,
      platforms: ['Twitter', 'Discord'],
      related_assets: ['ETH']
    },
    {
      hashtag: '#DogeArmy',
      mentions: 67800,
      sentiment: 45.8,
      change24h: -23.4,
      platforms: ['Twitter', 'Reddit'],
      related_assets: ['DOGE']
    }
  ]);

  const [discordChannels] = useState<DiscordChannel[]>([
    {
      name: 'alpha-calls',
      server: 'Crypto Titans',
      members: 45200,
      activity_score: 89.5,
      sentiment: 'bullish',
      recent_messages: 2340,
      top_discussed: ['BTC', 'ETH', 'SOL'],
      verified: true
    },
    {
      name: 'whale-alerts',
      server: 'DeFi Legends',
      members: 28900,
      activity_score: 92.1,
      sentiment: 'bullish',
      recent_messages: 1890,
      top_discussed: ['BTC', 'ETH'],
      verified: true
    },
    {
      name: 'meme-coins',
      server: 'Degen Paradise',
      members: 67800,
      activity_score: 78.4,
      sentiment: 'neutral',
      recent_messages: 4560,
      top_discussed: ['PEPE', 'DOGE', 'SHIB'],
      verified: false
    },
    {
      name: 'technical-analysis',
      server: 'Trading Academy',
      members: 34500,
      activity_score: 85.7,
      sentiment: 'bearish',
      recent_messages: 1234,
      top_discussed: ['BTC', 'ETH', 'SOL'],
      verified: true
    }
  ]);

  const getSentimentColor = (sentiment: string) => {
    switch (sentiment) {
      case 'bullish':
      case 'positive':
        return 'text-green-400';
      case 'bearish':
      case 'negative':
        return 'text-red-400';
      default:
        return 'text-gray-400';
    }
  };

  const getSentimentBadge = (sentiment: string) => {
    switch (sentiment) {
      case 'bullish':
      case 'positive':
        return 'bg-green-500/20 text-green-400 border-green-500/30';
      case 'bearish':
      case 'negative':
        return 'bg-red-500/20 text-red-400 border-red-500/30';
      default:
        return 'bg-gray-500/20 text-gray-400 border-gray-500/30';
    }
  };

  const getSentimentLabel = (sentiment: string) => {
    switch (sentiment) {
      case 'bullish':
        return 'صعودی';
      case 'bearish':
        return 'نزولی';
      case 'positive':
        return 'مثبت';
      case 'negative':
        return 'منفی';
      default:
        return 'خنثی';
    }
  };

  const getPlatformIcon = (platform: string) => {
    switch (platform) {
      case 'twitter': return Twitter;
      case 'discord': return Users;
      case 'lunar': return Crown;
      case 'reddit': return Globe;
      case 'telegram': return MessageSquare;
      default: return MessageSquare;
    }
  };

  const getPlatformColor = (platform: string) => {
    switch (platform) {
      case 'twitter': return 'text-blue-400';
      case 'discord': return 'text-purple-400';
      case 'lunar': return 'text-yellow-400';
      case 'reddit': return 'text-orange-400';
      case 'telegram': return 'text-cyan-400';
      default: return 'text-gray-400';
    }
  };

  const getPlatformLabel = (platform: string) => {
    switch (platform) {
      case 'twitter': return 'توییتر';
      case 'discord': return 'دیسکورد';
      case 'lunar': return 'Lunar';
      case 'reddit': return 'ردیت';
      case 'telegram': return 'تلگرام';
      default: return platform;
    }
  };

  const formatNumber = (num: number) => {
    if (num >= 1000000) return `${(num / 1000000).toFixed(1)}M`;
    if (num >= 1000) return `${(num / 1000).toFixed(1)}K`;
    return num.toString();
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="mb-6">
        <h1 className="text-3xl font-bold bg-gradient-to-r from-blue-400 to-purple-500 bg-clip-text text-transparent mb-2">
          مرکز هوش اجتماعی
        </h1>
        <p className="text-gray-400">سیگنال‌های اجتماعی بلادرنگ از توییتر/ایکس، دیسکورد، Lunar و موارد دیگر</p>
      </div>

      {/* Key Metrics Overview */}
      <div className="grid gap-4 md:grid-cols-4">
        <Card className="bg-gray-900/50 border-gray-700 backdrop-blur-sm">
          <CardHeader className="pb-2">
            <CardTitle className="text-white flex items-center gap-2">
              <Activity className="h-5 w-5 text-green-400" />
              احساسات بازار
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold text-green-400">صعودی</div>
            <p className="text-sm text-gray-400">۷۲.۴٪ سیگنال‌های مثبت</p>
          </CardContent>
        </Card>

        <Card className="bg-gray-900/50 border-gray-700 backdrop-blur-sm">
          <CardHeader className="pb-2">
            <CardTitle className="text-white flex items-center gap-2">
              <Hash className="h-5 w-5 text-blue-400" />
              کل اشاره‌ها
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold text-white">۱۴۲.۸K</div>
            <p className="text-sm text-gray-400">+۲۳۴٪ نسبت به دیروز</p>
          </CardContent>
        </Card>

        <Card className="bg-gray-900/50 border-gray-700 backdrop-blur-sm">
          <CardHeader className="pb-2">
            <CardTitle className="text-white flex items-center gap-2">
              <Flame className="h-5 w-5 text-orange-400" />
              پرطرفدار
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold text-white">#PepeSeason</div>
            <p className="text-sm text-gray-400">۵۶۷٪ رشد</p>
          </CardContent>
        </Card>

        <Card className="bg-gray-900/50 border-gray-700 backdrop-blur-sm">
          <CardHeader className="pb-2">
            <CardTitle className="text-white flex items-center gap-2">
              <Target className="h-5 w-5 text-purple-400" />
              امتیاز نفوذ
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold text-white">۸۴.۷</div>
            <p className="text-sm text-gray-400">سیگنال‌های با تأثیر بالا</p>
          </CardContent>
        </Card>
      </div>

      <Tabs defaultValue="sentiment" className="w-full">
        <TabsList className="grid w-full grid-cols-5 bg-gray-800 border-gray-700">
          <TabsTrigger value="sentiment" className="data-[state=active]:bg-blue-600">احساسات</TabsTrigger>
          <TabsTrigger value="influencers" className="data-[state=active]:bg-blue-600">تأثیرگذاران</TabsTrigger>
          <TabsTrigger value="trending" className="data-[state=active]:bg-blue-600">پرطرفدار</TabsTrigger>
          <TabsTrigger value="discord" className="data-[state=active]:bg-blue-600">دیسکورد</TabsTrigger>
          <TabsTrigger value="feed" className="data-[state=active]:bg-blue-600">فید زنده</TabsTrigger>
        </TabsList>

        <TabsContent value="sentiment" className="space-y-4">
          <Card className="bg-gray-900/50 border-gray-700 backdrop-blur-sm">
            <CardHeader>
              <CardTitle className="text-white flex items-center gap-2">
                <Activity className="h-5 w-5 text-blue-400" />
                تحلیل احساسات کریپتو
              </CardTitle>
              <CardDescription className="text-gray-400">
                احساسات بلادرنگ در تمام پلتفرم‌های اصلی
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="overflow-x-auto">
                <table className="w-full">
                  <thead>
                    <tr className="border-b border-gray-700">
                      <th className="text-left py-3 text-gray-300">دارایی</th>
                      <th className="text-left py-3 text-gray-300">احساسات</th>
                      <th className="text-right py-3 text-gray-300">امتیاز</th>
                      <th className="text-right py-3 text-gray-300">اشاره‌ها</th>
                      <th className="text-right py-3 text-gray-300">حجم</th>
                      <th className="text-right py-3 text-gray-300">نفوذ</th>
                      <th className="text-right py-3 text-gray-300">تغییر ۲۴ ساعته</th>
                    </tr>
                  </thead>
                  <tbody>
                    {sentimentData.map((item, index) => (
                      <tr key={index} className="border-b border-gray-800 hover:bg-gray-800/50">
                        <td className="py-3">
                          <div className="flex items-center gap-2">
                            <div className="w-8 h-8 rounded-full bg-gradient-to-r from-blue-500 to-purple-600 flex items-center justify-center text-white font-bold text-sm">
                              {item.symbol.slice(0, 2)}
                            </div>
                            <span className="font-medium text-white">{item.symbol}</span>
                          </div>
                        </td>
                        <td className="py-3">
                          <Badge className={getSentimentBadge(item.sentiment)}>
                            {getSentimentLabel(item.sentiment)}
                          </Badge>
                        </td>
                        <td className={`text-right py-3 font-bold ${getSentimentColor(item.sentiment)}`}>
                          {item.score.toFixed(1)}
                        </td>
                        <td className="text-right py-3 text-gray-300">{formatNumber(item.mentions)}</td>
                        <td className="text-right py-3 text-gray-300">{formatNumber(item.volume)}</td>
                        <td className="text-right py-3">
                          <div className="flex items-center justify-end gap-2">
                            <Progress value={item.influence} className="w-16 h-2" />
                            <span className="text-sm text-gray-300">{item.influence.toFixed(1)}</span>
                          </div>
                        </td>
                        <td className={`text-right py-3 font-medium ${item.change24h >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                          <div className="flex items-center justify-end gap-1">
                            {item.change24h >= 0 ? (
                              <TrendingUp className="h-4 w-4" />
                            ) : (
                              <TrendingDown className="h-4 w-4" />
                            )}
                            {item.change24h >= 0 ? '+' : ''}{item.change24h.toFixed(1)}%
                          </div>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </CardContent>
          </Card>

          {/* Platform Distribution */}
          <div className="grid gap-4 md:grid-cols-2">
            {sentimentData.slice(0, 2).map((item, index) => (
              <Card key={index} className="bg-gray-900/50 border-gray-700 backdrop-blur-sm">
                <CardHeader>
                  <CardTitle className="text-white text-lg">توزیع پلتفرم {item.symbol}</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="space-y-3">
                    {Object.entries(item.platforms).map(([platform, count]) => (
                      <div key={platform} className="flex items-center justify-between">
                        <div className="flex items-center gap-2">
                          <div className={`p-1 rounded ${getPlatformColor(platform)}`}>
                            {React.createElement(getPlatformIcon(platform), { className: "h-4 w-4" })}
                          </div>
                          <span className="text-gray-300">{getPlatformLabel(platform)}</span>
                        </div>
                        <div className="flex items-center gap-2">
                          <Progress value={(count / item.mentions) * 100} className="w-20 h-2" />
                          <span className="text-sm text-gray-400">{formatNumber(count)}</span>
                        </div>
                      </div>
                    ))}
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
        </TabsContent>

        <TabsContent value="influencers" className="space-y-4">
          <Card className="bg-gray-900/50 border-gray-700 backdrop-blur-sm">
            <CardHeader>
              <CardTitle className="text-white flex items-center gap-2">
                <Crown className="h-5 w-5 text-yellow-400" />
                برترین تأثیرگذاران کریپتو
              </CardTitle>
              <CardDescription className="text-gray-400">
                رهبران فکری کلیدی در شکل‌دهی احساسات بازار
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="grid gap-4 md:grid-cols-2">
                {influencers.map((influencer) => (
                  <div key={influencer.id} className="p-4 rounded-lg bg-gray-800/50 border border-gray-700">
                    <div className="flex items-center gap-3 mb-3">
                      <div className="text-2xl">{influencer.avatar}</div>
                      <div className="flex-1">
                        <div className="flex items-center gap-2">
                          <span className="font-medium text-white">{influencer.username}</span>
                          {influencer.verified && <CheckCircle className="h-4 w-4 text-blue-400" />}
                        </div>
                        <p className="text-sm text-gray-400">{influencer.platform}</p>
                      </div>
                      <Badge className={getSentimentBadge(influencer.recent_sentiment)}>
                        {getSentimentLabel(influencer.recent_sentiment)}
                      </Badge>
                    </div>

                    <div className="grid grid-cols-2 gap-4 text-sm">
                      <div>
                        <p className="text-gray-400">دنبال‌کنندگان</p>
                        <p className="text-white font-medium">{formatNumber(influencer.followers)}</p>
                      </div>
                      <div>
                        <p className="text-gray-400">امتیاز نفوذ</p>
                        <p className="text-white font-medium">{influencer.influence_score}</p>
                      </div>
                      <div>
                        <p className="text-gray-400">نرخ دقت</p>
                        <p className="text-white font-medium">{influencer.accuracy_rate}%</p>
                      </div>
                      <div>
                        <p className="text-gray-400">تخصص‌ها</p>
                        <div className="flex gap-1 flex-wrap">
                          {influencer.specialties.slice(0, 2).map((specialty) => (
                            <Badge key={specialty} variant="outline" className="text-xs">
                              {specialty}
                            </Badge>
                          ))}
                        </div>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="trending" className="space-y-4">
          <Card className="bg-gray-900/50 border-gray-700 backdrop-blur-sm">
            <CardHeader>
              <CardTitle className="text-white flex items-center gap-2">
                <Hash className="h-5 w-5 text-blue-400" />
                موضوعات پرطرفدار
              </CardTitle>
              <CardDescription className="text-gray-400">
                پرمناقشه‌ترین هشتگ‌ها و موضوعات در پلتفرم‌ها
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="grid gap-4 md:grid-cols-2">
                {trendingTopics.map((topic, index) => (
                  <div key={index} className="p-4 rounded-lg bg-gray-800/50 border border-gray-700">
                    <div className="flex items-center justify-between mb-3">
                      <div className="flex items-center gap-2">
                        <Hash className="h-4 w-4 text-blue-400" />
                        <span className="font-medium text-white">{topic.hashtag}</span>
                      </div>
                      <div className={`flex items-center gap-1 text-sm ${topic.change24h >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                        {topic.change24h >= 0 ? <TrendingUp className="h-3 w-3" /> : <TrendingDown className="h-3 w-3" />}
                        {topic.change24h >= 0 ? '+' : ''}{topic.change24h.toFixed(1)}%
                      </div>
                    </div>

                    <div className="space-y-2">
                      <div className="flex justify-between text-sm">
                        <span className="text-gray-400">اشاره‌ها</span>
                        <span className="text-white">{formatNumber(topic.mentions)}</span>
                      </div>
                      <div className="flex justify-between text-sm">
                        <span className="text-gray-400">احساسات</span>
                        <span className={getSentimentColor(topic.sentiment > 60 ? 'bullish' : topic.sentiment < 40 ? 'bearish' : 'neutral')}>
                          {topic.sentiment.toFixed(1)}
                        </span>
                      </div>
                      <div className="flex gap-1 flex-wrap">
                        {topic.platforms.map((platform) => (
                          <Badge key={platform} variant="outline" className="text-xs">
                            {platform}
                          </Badge>
                        ))}
                      </div>
                      <div className="flex gap-1 flex-wrap">
                        {topic.related_assets.map((asset) => (
                          <Badge key={asset} className="text-xs bg-blue-500/20 text-blue-400">
                            ${asset}
                          </Badge>
                        ))}
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="discord" className="space-y-4">
          <Card className="bg-gray-900/50 border-gray-700 backdrop-blur-sm">
            <CardHeader>
              <CardTitle className="text-white flex items-center gap-2">
                <Users className="h-5 w-5 text-purple-400" />
                کانال‌های فعال دیسکورد
              </CardTitle>
              <CardDescription className="text-gray-400">
                فعالیت بلادرنگ از برترین سرورهای کریپتوی دیسکورد
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="grid gap-4 md:grid-cols-2">
                {discordChannels.map((channel, index) => (
                  <div key={index} className="p-4 rounded-lg bg-gray-800/50 border border-gray-700">
                    <div className="flex items-center justify-between mb-3">
                      <div className="flex items-center gap-2">
                        <div className="w-2 h-2 rounded-full bg-green-400"></div>
                        <span className="font-medium text-white">#{channel.name}</span>
                        {channel.verified && <CheckCircle className="h-4 w-4 text-blue-400" />}
                      </div>
                      <Badge className={getSentimentBadge(channel.sentiment)}>
                        {getSentimentLabel(channel.sentiment)}
                      </Badge>
                    </div>

                    <p className="text-sm text-gray-400 mb-3">{channel.server}</p>

                    <div className="grid grid-cols-2 gap-4 text-sm mb-3">
                      <div>
                        <p className="text-gray-400">اعضا</p>
                        <p className="text-white font-medium">{formatNumber(channel.members)}</p>
                      </div>
                      <div>
                        <p className="text-gray-400">امتیاز فعالیت</p>
                        <p className="text-white font-medium">{channel.activity_score}</p>
                      </div>
                      <div>
                        <p className="text-gray-400">پیام‌های اخیر</p>
                        <p className="text-white font-medium">{formatNumber(channel.recent_messages)}</p>
                      </div>
                      <div>
                        <p className="text-gray-400">وضعیت</p>
                        <div className="flex items-center gap-1">
                          <div className="w-2 h-2 rounded-full bg-green-400"></div>
                          <span className="text-green-400 text-xs">فعال</span>
                        </div>
                      </div>
                    </div>

                    <div>
                      <p className="text-gray-400 text-sm mb-1">پرمناقشه‌ترین</p>
                      <div className="flex gap-1 flex-wrap">
                        {channel.top_discussed.map((asset) => (
                          <Badge key={asset} className="text-xs bg-purple-500/20 text-purple-400">
                            ${asset}
                          </Badge>
                        ))}
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="feed" className="space-y-4">
          <Card className="bg-gray-900/50 border-gray-700 backdrop-blur-sm">
            <CardHeader>
              <CardTitle className="text-white flex items-center gap-2">
                <MessageSquare className="h-5 w-5 text-green-400" />
                فید اجتماعی زنده
              </CardTitle>
              <CardDescription className="text-gray-400">
                پست‌های بلادرنگ از تمام پلتفرم‌ها
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                {socialPosts.map((post) => {
                  const PlatformIcon = getPlatformIcon(post.platform);
                  return (
                    <div key={post.id} className="p-4 rounded-lg bg-gray-800/50 border border-gray-700 hover:bg-gray-800/70 transition-colors">
                      <div className="flex items-start gap-3">
                        <div className={`p-2 rounded-lg ${getPlatformColor(post.platform)} bg-gray-700`}>
                          <PlatformIcon className="h-4 w-4" />
                        </div>
                        <div className="flex-1 min-w-0">
                          <div className="flex items-center gap-2 mb-2">
                            <span className="font-medium text-white">{post.author}</span>
                            {post.verified && <CheckCircle className="h-4 w-4 text-blue-400" />}
                            <Badge className={getSentimentBadge(post.sentiment)}>
                              {getSentimentLabel(post.sentiment)}
                            </Badge>
                            {post.channel && (
                              <Badge variant="outline" className="text-xs">
                                #{post.channel}
                              </Badge>
                            )}
                            <span className="text-sm text-gray-500 ml-auto">
                              <Clock className="h-3 w-3 inline mr-1" />
                              {formatDate(post.timestamp)}
                            </span>
                          </div>

                          <p className="text-gray-300 mb-3 whitespace-pre-line">{post.content}</p>

                          <div className="flex items-center justify-between">
                            <div className="flex items-center gap-4 text-sm text-gray-500">
                              <span className="flex items-center gap-1">
                                <Heart className="h-4 w-4" />
                                {formatNumber(post.engagement.likes)}
                              </span>
                              <span className="flex items-center gap-1">
                                <Share2 className="h-4 w-4" />
                                {formatNumber(post.engagement.retweets)}
                              </span>
                              <span className="flex items-center gap-1">
                                <MessageCircle className="h-4 w-4" />
                                {formatNumber(post.engagement.comments)}
                              </span>
                              <span className="flex items-center gap-1">
                                <Eye className="h-4 w-4" />
                                {formatNumber(post.engagement.views)}
                              </span>
                            </div>

                            <div className="flex items-center gap-1">
                              <Zap className="h-3 w-3 text-yellow-400" />
                              <span className="text-xs text-gray-400">
                                {post.influence_score.toFixed(1)}
                              </span>
                            </div>
                          </div>

                          <div className="flex gap-1 flex-wrap mt-2">
                            {post.mentions.map((mention) => (
                              <Badge key={mention} className="text-xs bg-green-500/20 text-green-400">
                                ${mention}
                              </Badge>
                            ))}
                            {post.tags.map((tag) => (
                              <Badge key={tag} variant="outline" className="text-xs">
                                #{tag}
                              </Badge>
                            ))}
                          </div>
                        </div>
                      </div>
                    </div>
                  );
                })}
              </div>

              <div className="text-center pt-4">
                <Button variant="outline" className="bg-gray-800 border-gray-600 text-white hover:bg-gray-700">
                  <ExternalLink className="h-4 w-4 mr-2" />
                  بارگذاری پست‌های بیشتر
                </Button>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        {/* Platform Sources Info */}
        <Card className="bg-gray-900/50 border-gray-700 backdrop-blur-sm mt-6">
          <CardHeader>
            <CardTitle className="text-white flex items-center gap-2">
              <Globe className="h-5 w-5 text-blue-400" />
              منابع داده متصل
            </CardTitle>
            <CardDescription className="text-gray-400">
              هوش اجتماعی بلادرنگ از پلتفرم‌های تأیید‌شده
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="grid gap-4 md:grid-cols-4">
              <div className="flex items-center justify-between p-4 rounded-lg bg-gray-800/50 border border-gray-700">
                <div className="flex items-center space-x-3">
                  <Twitter className="h-5 w-5 text-blue-400" />
                  <div>
                    <p className="font-medium text-white">توییتر/ایکس</p>
                    <p className="text-sm text-gray-400">توییت‌ها و ریسه‌های بلادرنگ</p>
                  </div>
                </div>
                <Badge className="bg-green-500/20 text-green-400">زنده</Badge>
              </div>

              <div className="flex items-center justify-between p-4 rounded-lg bg-gray-800/50 border border-gray-700">
                <div className="flex items-center space-x-3">
                  <Users className="h-5 w-5 text-purple-400" />
                  <div>
                    <p className="font-medium text-white">دیسکورد</p>
                    <p className="text-sm text-gray-400">کانال‌ها و پیام‌های سرور</p>
                  </div>
                </div>
                <Badge className="bg-green-500/20 text-green-400">زنده</Badge>
              </div>

              <div className="flex items-center justify-between p-4 rounded-lg bg-gray-800/50 border border-gray-700">
                <div className="flex items-center space-x-3">
                  <Crown className="h-5 w-5 text-yellow-400" />
                  <div>
                    <p className="font-medium text-white">Lunar</p>
                    <p className="text-sm text-gray-400">پلتفرم اجتماعی کریپتو</p>
                  </div>
                </div>
                <Badge className="bg-green-500/20 text-green-400">زنده</Badge>
              </div>

              <div className="flex items-center justify-between p-4 rounded-lg bg-gray-800/50 border border-gray-700">
                <div className="flex items-center space-x-3">
                  <Globe className="h-5 w-5 text-orange-400" />
                  <div>
                    <p className="font-medium text-white">ردیت</p>
                    <p className="text-sm text-gray-400">بحث‌های انجمن</p>
                  </div>
                </div>
                <Badge className="bg-green-500/20 text-green-400">زنده</Badge>
              </div>
            </div>
          </CardContent>
        </Card>
      </Tabs>
    </div>
  );
}
