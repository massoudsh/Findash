'use client';

import { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Input } from '@/components/ui/input';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { 
  HelpCircle,
  Search,
  Book,
  Video,
  MessageCircle,
  FileText,
  Download,
  ExternalLink,
  ChevronRight,
  ChevronDown,
  Star,
  Clock,
  Users,
  Zap,
  TrendingUp,
  DollarSign,
  BarChart3,
  AlertCircle,
  CheckCircle,
  PlayCircle,
  BookOpen,
  Phone,
  Mail,
  MessageSquare,
  Calendar,
  Globe,
  Code,
  PieChart,
  Monitor,
  Eye,
  Keyboard
} from 'lucide-react';

interface HelpArticle {
  id: string;
  title: string;
  category: string;
  difficulty: 'beginner' | 'intermediate' | 'advanced';
  readTime: number;
  rating: number;
  views: number;
  description: string;
  tags: string[];
  lastUpdated: string;
}

interface FAQ {
  id: string;
  question: string;
  answer: string;
  category: string;
  helpful: number;
  notHelpful: number;
}

interface Tutorial {
  id: string;
  title: string;
  description: string;
  duration: string;
  difficulty: 'beginner' | 'intermediate' | 'advanced';
  category: string;
  steps: number;
  completions: number;
  thumbnail: string;
}

export default function HelpPage() {
  const [selectedTab, setSelectedTab] = useState('overview');
  const [searchTerm, setSearchTerm] = useState('');
  const [selectedCategory, setSelectedCategory] = useState('all');
  const [expandedFAQ, setExpandedFAQ] = useState<string | null>(null);
  const [articles, setArticles] = useState<HelpArticle[]>([]);
  const [faqs, setFAQs] = useState<FAQ[]>([]);
  const [tutorials, setTutorials] = useState<Tutorial[]>([]);

  useEffect(() => {
    // Sample help articles
    const sampleArticles: HelpArticle[] = [
      {
        id: 'article-1',
        title: 'Getting Started with Octopus',
        category: 'Getting Started',
        difficulty: 'beginner',
        readTime: 5,
        rating: 4.8,
        views: 15420,
        description: 'Complete guide to setting up your account and making your first trade',
        tags: ['setup', 'basics', 'first-time'],
        lastUpdated: '2024-01-15T00:00:00Z'
      },
      {
        id: 'article-2',
        title: 'Understanding Risk Management',
        category: 'Risk Management',
        difficulty: 'intermediate',
        readTime: 12,
        rating: 4.9,
        views: 8930,
        description: 'Learn how to protect your investments with proper risk management strategies',
        tags: ['risk', 'portfolio', 'strategy'],
        lastUpdated: '2024-01-18T00:00:00Z'
      },
      {
        id: 'article-3',
        title: 'Advanced Options Trading Strategies',
        category: 'Options Trading',
        difficulty: 'advanced',
        readTime: 20,
        rating: 4.7,
        views: 3240,
        description: 'Deep dive into complex options strategies for experienced traders',
        tags: ['options', 'advanced', 'strategies'],
        lastUpdated: '2024-01-20T00:00:00Z'
      },
      {
        id: 'article-4',
        title: 'API Integration Guide',
        category: 'API & Development',
        difficulty: 'advanced',
        readTime: 25,
        rating: 4.6,
        views: 2180,
        description: 'Complete guide to integrating with our trading APIs',
        tags: ['api', 'development', 'integration'],
        lastUpdated: '2024-01-19T00:00:00Z'
      },
      {
        id: 'article-5',
        title: 'Portfolio Optimization Techniques',
        category: 'Portfolio Management',
        difficulty: 'intermediate',
        readTime: 15,
        rating: 4.8,
        views: 6750,
        description: 'Learn how to optimize your portfolio for maximum returns with minimal risk',
        tags: ['portfolio', 'optimization', 'returns'],
        lastUpdated: '2024-01-17T00:00:00Z'
      }
    ];

    const sampleFAQs: FAQ[] = [
      {
        id: 'faq-1',
        question: 'How do I start trading on the platform?',
        answer: 'To start trading with Octopus, first complete your account verification, deposit funds, and then navigate to the trading interface. You can place your first order by selecting an asset, choosing order type, and specifying quantity.',
        category: 'Getting Started',
        helpful: 245,
        notHelpful: 12
      },
      {
        id: 'faq-2',
        question: 'What are the minimum deposit requirements?',
        answer: 'The minimum deposit varies by account type: Basic accounts require $100, Premium accounts require $1,000, and Professional accounts require $10,000. There are no deposit fees for bank transfers.',
        category: 'Account & Billing',
        helpful: 189,
        notHelpful: 8
      },
      {
        id: 'faq-3',
        question: 'How does the risk management system work?',
        answer: 'Our risk management system monitors your positions in real-time, automatically calculating Value at Risk (VaR), position limits, and exposure limits. It can automatically close positions if risk thresholds are exceeded.',
        category: 'Risk Management',
        helpful: 156,
        notHelpful: 23
      },
      {
        id: 'faq-4',
        question: 'Can I use the API for automated trading?',
        answer: 'Yes, we provide comprehensive REST and WebSocket APIs for automated trading. You can access real-time market data, place orders, manage positions, and retrieve account information programmatically.',
        category: 'API & Technical',
        helpful: 98,
        notHelpful: 5
      }
    ];

    const sampleTutorials: Tutorial[] = [
      {
        id: 'tutorial-1',
        title: 'Your First Trade',
        description: 'Step-by-step guide to placing your first order',
        duration: '8 mins',
        difficulty: 'beginner',
        category: 'Getting Started',
        steps: 6,
        completions: 12840,
        thumbnail: '/tutorial-first-trade.jpg'
      },
      {
        id: 'tutorial-2',
        title: 'Setting Up Stop Losses',
        description: 'Learn how to protect your investments with stop losses',
        duration: '12 mins',
        difficulty: 'beginner',
        category: 'Risk Management',
        steps: 8,
        completions: 8650,
        thumbnail: '/tutorial-stop-loss.jpg'
      },
      {
        id: 'tutorial-3',
        title: 'Options Trading Basics',
        description: 'Introduction to buying and selling options',
        duration: '25 mins',
        difficulty: 'intermediate',
        category: 'Options Trading',
        steps: 12,
        completions: 4230,
        thumbnail: '/tutorial-options.jpg'
      }
    ];

    setArticles(sampleArticles);
    setFAQs(sampleFAQs);
    setTutorials(sampleTutorials);
  }, []);

  const categories = [
    'all',
    'Getting Started',
    'Trading',
    'Portfolio Management',
    'Options Trading',
    'Risk Management',
    'API & Development',
    'Account & Billing'
  ];

  const filteredArticles = articles.filter(article => {
    const matchesSearch = article.title.toLowerCase().includes(searchTerm.toLowerCase()) ||
                         article.description.toLowerCase().includes(searchTerm.toLowerCase()) ||
                         article.tags.some(tag => tag.toLowerCase().includes(searchTerm.toLowerCase()));
    const matchesCategory = selectedCategory === 'all' || article.category === selectedCategory;
    return matchesSearch && matchesCategory;
  });

  const filteredFAQs = faqs.filter(faq => {
    const matchesSearch = faq.question.toLowerCase().includes(searchTerm.toLowerCase()) ||
                         faq.answer.toLowerCase().includes(searchTerm.toLowerCase());
    const matchesCategory = selectedCategory === 'all' || faq.category === selectedCategory;
    return matchesSearch && matchesCategory;
  });

  const filteredTutorials = tutorials.filter(tutorial => {
    const matchesSearch = tutorial.title.toLowerCase().includes(searchTerm.toLowerCase()) ||
                         tutorial.description.toLowerCase().includes(searchTerm.toLowerCase());
    const matchesCategory = selectedCategory === 'all' || tutorial.category === selectedCategory;
    return matchesSearch && matchesCategory;
  });

  const getDifficultyColor = (difficulty: string) => {
    switch (difficulty) {
      case 'beginner': return 'bg-green-100 text-green-800';
      case 'intermediate': return 'bg-yellow-100 text-yellow-800';
      case 'advanced': return 'bg-red-100 text-red-800';
      default: return 'bg-gray-100 text-gray-800';
    }
  };

  const formatDate = (dateString: string) => {
    return new Date(dateString).toLocaleDateString();
  };

  return (
    <div className="space-y-6">
      <div className="flex justify-between items-start">
        <div>
          <h1 className="text-3xl font-bold tracking-tight flex items-center gap-3">
            <HelpCircle className="w-8 h-8 text-blue-600" />
            Help & Documentation
          </h1>
          <p className="text-muted-foreground">
            Comprehensive guides, tutorials, and support resources for Octopus
          </p>
        </div>
        
        <div className="flex gap-3">
          <Button variant="outline" className="flex items-center gap-2">
            <MessageCircle className="w-4 h-4" />
            Contact Support
          </Button>
          <Button className="flex items-center gap-2">
            <Video className="w-4 h-4" />
            Watch Demo
          </Button>
        </div>
      </div>

      <Card>
        <CardContent className="p-6">
          <div className="flex gap-4 items-center">
            <div className="flex-1 relative">
              <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400 w-4 h-4" />
              <Input
                placeholder="Search help articles, FAQs, and tutorials..."
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
                className="pl-10"
              />
            </div>
            <select 
              value={selectedCategory} 
              onChange={(e) => setSelectedCategory(e.target.value)}
              className="px-3 py-2 border rounded-md min-w-48"
            >
              {categories.map(category => (
                <option key={category} value={category}>
                  {category === 'all' ? 'All Categories' : category}
                </option>
              ))}
            </select>
          </div>
        </CardContent>
      </Card>

      <Tabs value={selectedTab} onValueChange={setSelectedTab} className="space-y-6">
        <TabsList className="grid w-full grid-cols-6">
          <TabsTrigger value="overview">Overview</TabsTrigger>
          <TabsTrigger value="guides">User Guides</TabsTrigger>
          <TabsTrigger value="tutorials">Tutorials</TabsTrigger>
          <TabsTrigger value="faq">FAQ</TabsTrigger>
          <TabsTrigger value="api">API Docs</TabsTrigger>
          <TabsTrigger value="support">Support</TabsTrigger>
        </TabsList>

        <TabsContent value="overview" className="space-y-6">
          <div className="grid gap-6 md:grid-cols-3">
            <Card className="cursor-pointer hover:shadow-lg transition-shadow" onClick={() => setSelectedTab('guides')}>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Book className="w-5 h-5 text-blue-600" />
                  User Guides
                </CardTitle>
              </CardHeader>
              <CardContent>
                <p className="text-sm text-gray-600 mb-4">
                  Comprehensive documentation and step-by-step guides
                </p>
                <div className="flex items-center justify-between">
                  <Badge variant="outline">{articles.length} articles</Badge>
                  <ChevronRight className="w-4 h-4 text-gray-400" />
                </div>
              </CardContent>
            </Card>

            <Card className="cursor-pointer hover:shadow-lg transition-shadow" onClick={() => setSelectedTab('tutorials')}>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <PlayCircle className="w-5 h-5 text-green-600" />
                  Video Tutorials
                </CardTitle>
              </CardHeader>
              <CardContent>
                <p className="text-sm text-gray-600 mb-4">
                  Interactive video tutorials and walkthroughs
                </p>
                <div className="flex items-center justify-between">
                  <Badge variant="outline">{tutorials.length} tutorials</Badge>
                  <ChevronRight className="w-4 h-4 text-gray-400" />
                </div>
              </CardContent>
            </Card>

            <Card className="cursor-pointer hover:shadow-lg transition-shadow" onClick={() => setSelectedTab('faq')}>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <MessageSquare className="w-5 h-5 text-purple-600" />
                  Frequently Asked
                </CardTitle>
              </CardHeader>
              <CardContent>
                <p className="text-sm text-gray-600 mb-4">
                  Quick answers to common questions
                </p>
                <div className="flex items-center justify-between">
                  <Badge variant="outline">{faqs.length} questions</Badge>
                  <ChevronRight className="w-4 h-4 text-gray-400" />
                </div>
              </CardContent>
            </Card>
          </div>

          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <TrendingUp className="w-5 h-5" />
                Most Popular Articles
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-3">
                {articles.slice(0, 3).map((article, index) => (
                  <div key={article.id} className="flex items-center gap-3 p-3 border rounded-lg hover:bg-gray-50 cursor-pointer">
                    <div className="w-6 h-6 bg-blue-100 rounded-full flex items-center justify-center text-sm font-medium text-blue-600">
                      {index + 1}
                    </div>
                    <div className="flex-1">
                      <div className="font-medium">{article.title}</div>
                      <div className="text-sm text-gray-500">{article.description}</div>
                    </div>
                    <div className="flex items-center gap-2 text-sm text-gray-500">
                      <Eye className="w-4 h-4" />
                      {article.views.toLocaleString()}
                    </div>
                    <Badge className={getDifficultyColor(article.difficulty)}>
                      {article.difficulty}
                    </Badge>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>

          <div className="grid gap-6 md:grid-cols-2">
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Zap className="w-5 h-5" />
                  Quick Access
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-3">
                <Button variant="outline" className="w-full justify-start">
                  <PlayCircle className="w-4 h-4 mr-2" />
                  Platform Overview Video
                </Button>
                <Button variant="outline" className="w-full justify-start">
                  <Download className="w-4 h-4 mr-2" />
                  Download User Manual
                </Button>
                <Button variant="outline" className="w-full justify-start">
                  <Code className="w-4 h-4 mr-2" />
                  API Reference
                </Button>
                <Button variant="outline" className="w-full justify-start">
                  <Keyboard className="w-4 h-4 mr-2" />
                  Keyboard Shortcuts
                </Button>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <MessageCircle className="w-5 h-5" />
                  Need Help?
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-3">
                <Button className="w-full justify-start">
                  <Mail className="w-4 h-4 mr-2" />
                  Email Support
                </Button>
                <Button variant="outline" className="w-full justify-start">
                  <Phone className="w-4 h-4 mr-2" />
                  Call Support
                </Button>
                <Button variant="outline" className="w-full justify-start">
                  <MessageSquare className="w-4 h-4 mr-2" />
                  Live Chat
                </Button>
                <Button variant="outline" className="w-full justify-start">
                  <Calendar className="w-4 h-4 mr-2" />
                  Schedule Demo
                </Button>
              </CardContent>
            </Card>
          </div>
        </TabsContent>

        <TabsContent value="guides" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <BookOpen className="w-5 h-5" />
                User Guides & Documentation
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="grid gap-4">
                {filteredArticles.map(article => (
                  <div key={article.id} className="border rounded-lg p-4 hover:bg-gray-50 cursor-pointer">
                    <div className="flex items-start justify-between mb-3">
                      <div className="flex-1">
                        <h3 className="font-medium mb-1">{article.title}</h3>
                        <p className="text-sm text-gray-600 mb-2">{article.description}</p>
                        <div className="flex items-center gap-3 text-xs text-gray-500">
                          <div className="flex items-center gap-1">
                            <Clock className="w-3 h-3" />
                            {article.readTime} min read
                          </div>
                          <div className="flex items-center gap-1">
                            <Star className="w-3 h-3" />
                            {article.rating}
                          </div>
                          <div className="flex items-center gap-1">
                            <Eye className="w-3 h-3" />
                            {article.views.toLocaleString()} views
                          </div>
                          <span>Updated {formatDate(article.lastUpdated)}</span>
                        </div>
                      </div>
                      <div className="flex items-center gap-2">
                        <Badge className={getDifficultyColor(article.difficulty)}>
                          {article.difficulty}
                        </Badge>
                        <Badge variant="outline">{article.category}</Badge>
                      </div>
                    </div>
                    <div className="flex items-center justify-between">
                      <div className="flex gap-1">
                        {article.tags.map(tag => (
                          <Badge key={tag} variant="outline" className="text-xs">
                            {tag}
                          </Badge>
                        ))}
                      </div>
                      <Button variant="outline" size="sm">
                        Read Article
                      </Button>
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="tutorials" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Video className="w-5 h-5" />
                Interactive Tutorials
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="grid gap-4 md:grid-cols-2">
                {filteredTutorials.map(tutorial => (
                  <div key={tutorial.id} className="border rounded-lg p-4 hover:bg-gray-50 cursor-pointer">
                    <div className="aspect-video bg-gray-200 rounded-lg mb-3 flex items-center justify-center">
                      <PlayCircle className="w-12 h-12 text-gray-400" />
                    </div>
                    <div className="space-y-3">
                      <div>
                        <h3 className="font-medium mb-1">{tutorial.title}</h3>
                        <p className="text-sm text-gray-600">{tutorial.description}</p>
                      </div>
                      <div className="flex items-center justify-between text-sm">
                        <div className="flex items-center gap-3 text-gray-500">
                          <span>{tutorial.duration}</span>
                          <span>{tutorial.steps} steps</span>
                          <span>{tutorial.completions.toLocaleString()} completed</span>
                        </div>
                        <Badge className={getDifficultyColor(tutorial.difficulty)}>
                          {tutorial.difficulty}
                        </Badge>
                      </div>
                      <Button className="w-full">
                        <PlayCircle className="w-4 h-4 mr-2" />
                        Start Tutorial
                      </Button>
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="faq" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <HelpCircle className="w-5 h-5" />
                Frequently Asked Questions
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-3">
                {filteredFAQs.map(faq => (
                  <div key={faq.id} className="border rounded-lg">
                    <button
                      className="w-full p-4 text-left flex items-center justify-between hover:bg-gray-50"
                      onClick={() => setExpandedFAQ(expandedFAQ === faq.id ? null : faq.id)}
                    >
                      <span className="font-medium">{faq.question}</span>
                      {expandedFAQ === faq.id ? 
                        <ChevronDown className="w-4 h-4" /> : 
                        <ChevronRight className="w-4 h-4" />
                      }
                    </button>
                    {expandedFAQ === faq.id && (
                      <div className="p-4 border-t bg-gray-50">
                        <p className="text-gray-700 mb-4">{faq.answer}</p>
                        <div className="flex items-center justify-between">
                          <div className="flex items-center gap-4">
                            <span className="text-sm text-gray-500">Was this helpful?</span>
                            <div className="flex gap-2">
                              <Button variant="outline" size="sm">
                                <CheckCircle className="w-4 h-4 mr-1" />
                                Yes ({faq.helpful})
                              </Button>
                              <Button variant="outline" size="sm">
                                <AlertCircle className="w-4 h-4 mr-1" />
                                No ({faq.notHelpful})
                              </Button>
                            </div>
                          </div>
                          <Badge variant="outline">{faq.category}</Badge>
                        </div>
                      </div>
                    )}
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="api" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Code className="w-5 h-5" />
                API Documentation
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-6">
              <div className="grid gap-4 md:grid-cols-2">
                <div className="space-y-4">
                  <h3 className="font-medium">REST API</h3>
                  <div className="space-y-2">
                    <Button variant="outline" className="w-full justify-start">
                      <FileText className="w-4 h-4 mr-2" />
                      Authentication
                    </Button>
                    <Button variant="outline" className="w-full justify-start">
                      <DollarSign className="w-4 h-4 mr-2" />
                      Trading Endpoints
                    </Button>
                    <Button variant="outline" className="w-full justify-start">
                      <PieChart className="w-4 h-4 mr-2" />
                      Portfolio Management
                    </Button>
                    <Button variant="outline" className="w-full justify-start">
                      <BarChart3 className="w-4 h-4 mr-2" />
                      Market Data
                    </Button>
                  </div>
                </div>
                <div className="space-y-4">
                  <h3 className="font-medium">WebSocket API</h3>
                  <div className="space-y-2">
                    <Button variant="outline" className="w-full justify-start">
                      <Zap className="w-4 h-4 mr-2" />
                      Real-time Feeds
                    </Button>
                    <Button variant="outline" className="w-full justify-start">
                      <Monitor className="w-4 h-4 mr-2" />
                      Order Updates
                    </Button>
                    <Button variant="outline" className="w-full justify-start">
                      <Users className="w-4 h-4 mr-2" />
                      Account Events
                    </Button>
                    <Button variant="outline" className="w-full justify-start">
                      <AlertCircle className="w-4 h-4 mr-2" />
                      Error Handling
                    </Button>
                  </div>
                </div>
              </div>
              
              <div className="grid gap-4 md:grid-cols-3">
                <Button className="flex items-center gap-2">
                  <Download className="w-4 h-4" />
                  Download SDK
                </Button>
                <Button variant="outline" className="flex items-center gap-2">
                  <ExternalLink className="w-4 h-4" />
                  Interactive API Explorer
                </Button>
                <Button variant="outline" className="flex items-center gap-2">
                  <Code className="w-4 h-4" />
                  Code Examples
                </Button>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="support" className="space-y-4">
          <div className="grid gap-6 md:grid-cols-2">
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <MessageCircle className="w-5 h-5" />
                  Contact Support
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="space-y-3">
                  <Button className="w-full justify-start">
                    <MessageSquare className="w-4 h-4 mr-2" />
                    Live Chat (24/7)
                  </Button>
                  <Button variant="outline" className="w-full justify-start">
                    <Mail className="w-4 h-4 mr-2" />
                    Email Support
                  </Button>
                  <Button variant="outline" className="w-full justify-start">
                    <Phone className="w-4 h-4 mr-2" />
                    Phone Support
                  </Button>
                  <Button variant="outline" className="w-full justify-start">
                    <Calendar className="w-4 h-4 mr-2" />
                    Schedule Consultation
                  </Button>
                </div>
                
                <div className="border-t pt-4">
                  <h4 className="font-medium mb-2">Support Hours</h4>
                  <div className="space-y-1 text-sm text-gray-600">
                    <div className="flex justify-between">
                      <span>Live Chat:</span>
                      <span>24/7</span>
                    </div>
                    <div className="flex justify-between">
                      <span>Phone Support:</span>
                      <span>Mon-Fri 8AM-8PM EST</span>
                    </div>
                    <div className="flex justify-between">
                      <span>Email Response:</span>
                      <span>Within 2 hours</span>
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Globe className="w-5 h-5" />
                  Community & Resources
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="space-y-3">
                  <Button variant="outline" className="w-full justify-start">
                    <Users className="w-4 h-4 mr-2" />
                    Community Forum
                  </Button>
                  <Button variant="outline" className="w-full justify-start">
                    <ExternalLink className="w-4 h-4 mr-2" />
                    Blog & Updates
                  </Button>
                  <Button variant="outline" className="w-full justify-start">
                    <Video className="w-4 h-4 mr-2" />
                    Webinar Library
                  </Button>
                  <Button variant="outline" className="w-full justify-start">
                    <FileText className="w-4 h-4 mr-2" />
                    White Papers
                  </Button>
                </div>

                <div className="border-t pt-4">
                  <h4 className="font-medium mb-2">System Status</h4>
                  <div className="flex items-center justify-between">
                    <span className="text-sm">All Systems Operational</span>
                    <div className="flex items-center gap-1">
                      <div className="w-2 h-2 bg-green-500 rounded-full"></div>
                      <span className="text-xs text-green-600">Online</span>
                    </div>
                  </div>
                  <Button variant="outline" size="sm" className="w-full mt-2">
                    <Monitor className="w-3 h-3 mr-1" />
                    Status Page
                  </Button>
                </div>
              </CardContent>
            </Card>
          </div>
        </TabsContent>
      </Tabs>
    </div>
  );
} 