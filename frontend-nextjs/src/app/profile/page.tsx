'use client';

import { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Input } from '@/components/ui/input';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { 
  User,
  Settings,
  Shield,
  Key,
  Bell,
  CreditCard,
  Eye,
  EyeOff,
  Edit,
  Save,
  X,
  Check,
  AlertTriangle,
  Download,
  Upload,
  Smartphone,
  Mail,
  Phone,
  MapPin,
  Calendar,
  Clock,
  Globe,
  Lock,
  Unlock,
  Copy,
  RefreshCw,
  Trash2,
  Plus,
  Activity,
  BarChart3,
  DollarSign,
  TrendingUp,
  Zap,
  Target,
  Camera
} from 'lucide-react';

interface UserProfile {
  id: string;
  firstName: string;
  lastName: string;
  email: string;
  phone: string;
  avatar: string;
  role: string;
  joinDate: string;
  lastLogin: string;
  location: string;
  timezone: string;
  bio: string;
  verified: boolean;
  twoFactorEnabled: boolean;
  emailNotifications: boolean;
  smsNotifications: boolean;
  pushNotifications: boolean;
}

interface TradingPreferences {
  defaultOrderType: string;
  defaultTimeInForce: string;
  riskTolerance: string;
  maxPositionSize: number;
  stopLossDefault: number;
  takeProfitDefault: number;
  autoConfirmOrders: boolean;
  showAdvancedFeatures: boolean;
  preferredMarkets: string[];
  tradingHours: { start: string; end: string };
}

interface APIKey {
  id: string;
  name: string;
  key: string;
  permissions: string[];
  created: string;
  lastUsed: string;
  status: 'active' | 'inactive';
}

interface SecurityLog {
  id: string;
  action: string;
  timestamp: string;
  ipAddress: string;
  location: string;
  device: string;
  status: 'success' | 'failed';
}

export default function ProfilePage() {
  const [selectedTab, setSelectedTab] = useState('personal');
  const [isEditing, setIsEditing] = useState(false);
  const [showApiKey, setShowApiKey] = useState<string | null>(null);
  
  const [profile, setProfile] = useState<UserProfile>({
    id: 'user-123',
    firstName: 'Massoud',
    lastName: 'Shemirani',
    email: 'massoud.shemirani@company.com',
    phone: '+1 (555) 123-4567',
    avatar: '',
    role: 'Senior Trader',
    joinDate: '2022-03-15',
    lastLogin: '2024-01-20T15:45:23Z',
    location: 'New York, NY, USA',
    timezone: 'EST (UTC-5)',
    bio: 'Experienced quantitative trader specializing in algorithmic strategies and risk management.',
    verified: true,
    twoFactorEnabled: true,
    emailNotifications: true,
    smsNotifications: false,
    pushNotifications: true
  });

  const [tradingPrefs, setTradingPrefs] = useState<TradingPreferences>({
    defaultOrderType: 'limit',
    defaultTimeInForce: 'day',
    riskTolerance: 'moderate',
    maxPositionSize: 100000,
    stopLossDefault: 2.0,
    takeProfitDefault: 4.0,
    autoConfirmOrders: false,
    showAdvancedFeatures: true,
    preferredMarkets: ['stocks', 'crypto', 'forex'],
    tradingHours: { start: '09:30', end: '16:00' }
  });

  const [apiKeys, setApiKeys] = useState<APIKey[]>([
    {
      id: 'api-001',
      name: 'Trading Bot v1',
      key: 'ak_live_123abc...def789',
      permissions: ['read', 'trade'],
      created: '2024-01-15',
      lastUsed: '2024-01-20T10:30:00Z',
      status: 'active'
    },
    {
      id: 'api-002',
      name: 'Data Analytics',
      key: 'ak_live_456ghi...jkl012',
      permissions: ['read'],
      created: '2024-01-10',
      lastUsed: '2024-01-19T14:15:00Z',
      status: 'active'
    }
  ]);

  const [securityLogs, setSecurityLogs] = useState<SecurityLog[]>([
    {
      id: 'sec-001',
      action: 'Login',
      timestamp: '2024-01-20T15:45:23Z',
      ipAddress: '192.168.1.145',
      location: 'New York, NY',
      device: 'Chrome on Windows',
      status: 'success'
    },
    {
      id: 'sec-002',
      action: 'Password Change',
      timestamp: '2024-01-18T09:30:15Z',
      ipAddress: '192.168.1.145',
      location: 'New York, NY',
      device: 'Chrome on Windows',
      status: 'success'
    },
    {
      id: 'sec-003',
      action: 'Failed Login',
      timestamp: '2024-01-17T23:15:42Z',
      ipAddress: '203.45.67.89',
      location: 'Unknown',
      device: 'Unknown',
      status: 'failed'
    }
  ]);

  const handleSaveProfile = () => {
    // Save profile logic
    setIsEditing(false);
  };

  const handleGenerateApiKey = () => {
    const newKey: APIKey = {
      id: `api-${Date.now()}`,
      name: `New API Key ${apiKeys.length + 1}`,
      key: `ak_live_${Math.random().toString(36).substring(2, 15)}...`,
      permissions: ['read'],
      created: new Date().toISOString().split('T')[0],
      lastUsed: 'Never',
      status: 'active'
    };
    setApiKeys([...apiKeys, newKey]);
  };

  const copyToClipboard = (text: string) => {
    navigator.clipboard.writeText(text);
  };

  const formatTimestamp = (timestamp: string) => {
    return new Date(timestamp).toLocaleString();
  };

  return (
    <div className="space-y-6">
      <div className="flex justify-between items-start">
        <div>
          <h1 className="text-3xl font-bold tracking-tight flex items-center gap-3">
            <User className="w-8 h-8 text-blue-600" />
            User Profile
          </h1>
          <p className="text-muted-foreground">
            Manage your account settings, security, and trading preferences
          </p>
        </div>
        
        <div className="flex gap-3">
          {isEditing ? (
            <>
              <Button variant="outline" onClick={() => setIsEditing(false)}>
                <X className="w-4 h-4 mr-2" />
                Cancel
              </Button>
              <Button onClick={handleSaveProfile}>
                <Save className="w-4 h-4 mr-2" />
                Save Changes
              </Button>
            </>
          ) : (
            <Button onClick={() => setIsEditing(true)}>
              <Edit className="w-4 h-4 mr-2" />
              Edit Profile
            </Button>
          )}
        </div>
      </div>

      {/* Profile Header */}
      <Card>
        <CardContent className="p-6">
          <div className="flex items-center gap-6">
            <div className="relative">
              <div className="w-24 h-24 bg-gradient-to-br from-blue-500 to-purple-600 rounded-full flex items-center justify-center text-white text-2xl font-bold">
                {profile.firstName[0]}{profile.lastName[0]}
              </div>
              {isEditing && (
                <Button size="sm" className="absolute -bottom-2 -right-2 rounded-full w-8 h-8 p-0">
                  <Camera className="w-4 h-4" />
                </Button>
              )}
            </div>
            
            <div className="flex-1">
              <div className="flex items-center gap-3 mb-2">
                <h2 className="text-2xl font-bold">{profile.firstName} {profile.lastName}</h2>
                {profile.verified && (
                  <Badge className="bg-green-100 text-green-800">
                    <Check className="w-3 h-3 mr-1" />
                    Verified
                  </Badge>
                )}
                <Badge variant="outline">{profile.role}</Badge>
              </div>
              <p className="text-gray-600 mb-2">{profile.email}</p>
              <div className="flex items-center gap-4 text-sm text-gray-500">
                <span className="flex items-center gap-1">
                  <Calendar className="w-4 h-4" />
                  Joined {new Date(profile.joinDate).toLocaleDateString()}
                </span>
                <span className="flex items-center gap-1">
                  <Clock className="w-4 h-4" />
                  Last login {formatTimestamp(profile.lastLogin)}
                </span>
                <span className="flex items-center gap-1">
                  <MapPin className="w-4 h-4" />
                  {profile.location}
                </span>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>

      <Tabs value={selectedTab} onValueChange={setSelectedTab} className="space-y-6">
        <TabsList className="grid w-full grid-cols-5">
          <TabsTrigger value="personal">Personal Info</TabsTrigger>
          <TabsTrigger value="security">Security</TabsTrigger>
          <TabsTrigger value="trading">Trading</TabsTrigger>
          <TabsTrigger value="api">API Keys</TabsTrigger>
          <TabsTrigger value="notifications">Notifications</TabsTrigger>
        </TabsList>

        <TabsContent value="personal" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Personal Information</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="grid gap-4 md:grid-cols-2">
                <div>
                  <label className="text-sm font-medium">First Name</label>
                  <Input 
                    value={profile.firstName} 
                    onChange={(e) => setProfile({...profile, firstName: e.target.value})}
                    disabled={!isEditing}
                  />
                </div>
                <div>
                  <label className="text-sm font-medium">Last Name</label>
                  <Input 
                    value={profile.lastName} 
                    onChange={(e) => setProfile({...profile, lastName: e.target.value})}
                    disabled={!isEditing}
                  />
                </div>
              </div>
              
              <div>
                <label className="text-sm font-medium">Email Address</label>
                <Input 
                  value={profile.email} 
                  onChange={(e) => setProfile({...profile, email: e.target.value})}
                  disabled={!isEditing}
                />
              </div>
              
              <div className="grid gap-4 md:grid-cols-2">
                <div>
                  <label className="text-sm font-medium">Phone Number</label>
                  <Input 
                    value={profile.phone} 
                    onChange={(e) => setProfile({...profile, phone: e.target.value})}
                    disabled={!isEditing}
                  />
                </div>
                <div>
                  <label className="text-sm font-medium">Timezone</label>
                  <Input 
                    value={profile.timezone} 
                    onChange={(e) => setProfile({...profile, timezone: e.target.value})}
                    disabled={!isEditing}
                  />
                </div>
              </div>
              
              <div>
                <label className="text-sm font-medium">Bio</label>
                <textarea 
                  className="w-full p-3 border rounded-md resize-none"
                  rows={3}
                  value={profile.bio}
                  onChange={(e) => setProfile({...profile, bio: e.target.value})}
                  disabled={!isEditing}
                />
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="security" className="space-y-4">
          <div className="grid gap-4 md:grid-cols-2">
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Shield className="w-5 h-5" />
                  Security Settings
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="flex items-center justify-between p-3 border rounded-lg">
                  <div>
                    <div className="font-medium">Two-Factor Authentication</div>
                    <div className="text-sm text-gray-600">Add an extra layer of security</div>
                  </div>
                  <div className="flex items-center gap-2">
                    {profile.twoFactorEnabled ? (
                      <Badge className="bg-green-100 text-green-800">Enabled</Badge>
                    ) : (
                      <Badge variant="outline">Disabled</Badge>
                    )}
                    <Button size="sm" variant="outline">
                      {profile.twoFactorEnabled ? 'Disable' : 'Enable'}
                    </Button>
                  </div>
                </div>
                
                <div className="flex items-center justify-between p-3 border rounded-lg">
                  <div>
                    <div className="font-medium">Password</div>
                    <div className="text-sm text-gray-600">Last changed 3 days ago</div>
                  </div>
                  <Button size="sm" variant="outline">
                    Change Password
                  </Button>
                </div>
                
                <div className="flex items-center justify-between p-3 border rounded-lg">
                  <div>
                    <div className="font-medium">Login Sessions</div>
                    <div className="text-sm text-gray-600">Manage active sessions</div>
                  </div>
                  <Button size="sm" variant="outline">
                    View Sessions
                  </Button>
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Activity className="w-5 h-5" />
                  Recent Activity
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-3">
                  {securityLogs.slice(0, 4).map((log) => (
                    <div key={log.id} className="flex items-center justify-between p-2 border rounded">
                      <div className="flex items-center gap-3">
                        <div className={`w-2 h-2 rounded-full ${
                          log.status === 'success' ? 'bg-green-500' : 'bg-red-500'
                        }`} />
                        <div>
                          <div className="text-sm font-medium">{log.action}</div>
                          <div className="text-xs text-gray-600">
                            {formatTimestamp(log.timestamp)} • {log.location}
                          </div>
                        </div>
                      </div>
                      <Badge className={log.status === 'success' ? 'bg-green-100 text-green-800' : 'bg-red-100 text-red-800'}>
                        {log.status}
                      </Badge>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          </div>
        </TabsContent>

        <TabsContent value="trading" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <TrendingUp className="w-5 h-5" />
                Trading Preferences
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="grid gap-4 md:grid-cols-2">
                <div>
                  <label className="text-sm font-medium">Default Order Type</label>
                  <select 
                    value={tradingPrefs.defaultOrderType}
                    onChange={(e) => setTradingPrefs({...tradingPrefs, defaultOrderType: e.target.value})}
                    className="w-full px-3 py-2 border rounded-md"
                  >
                    <option value="market">Market</option>
                    <option value="limit">Limit</option>
                    <option value="stop">Stop</option>
                    <option value="stop-limit">Stop Limit</option>
                  </select>
                </div>
                
                <div>
                  <label className="text-sm font-medium">Risk Tolerance</label>
                  <select 
                    value={tradingPrefs.riskTolerance}
                    onChange={(e) => setTradingPrefs({...tradingPrefs, riskTolerance: e.target.value})}
                    className="w-full px-3 py-2 border rounded-md"
                  >
                    <option value="conservative">Conservative</option>
                    <option value="moderate">Moderate</option>
                    <option value="aggressive">Aggressive</option>
                  </select>
                </div>
              </div>
              
              <div className="grid gap-4 md:grid-cols-3">
                <div>
                  <label className="text-sm font-medium">Max Position Size ($)</label>
                  <Input 
                    type="number"
                    value={tradingPrefs.maxPositionSize}
                    onChange={(e) => setTradingPrefs({...tradingPrefs, maxPositionSize: Number(e.target.value)})}
                  />
                </div>
                <div>
                  <label className="text-sm font-medium">Default Stop Loss (%)</label>
                  <Input 
                    type="number"
                    step="0.1"
                    value={tradingPrefs.stopLossDefault}
                    onChange={(e) => setTradingPrefs({...tradingPrefs, stopLossDefault: Number(e.target.value)})}
                  />
                </div>
                <div>
                  <label className="text-sm font-medium">Default Take Profit (%)</label>
                  <Input 
                    type="number"
                    step="0.1"
                    value={tradingPrefs.takeProfitDefault}
                    onChange={(e) => setTradingPrefs({...tradingPrefs, takeProfitDefault: Number(e.target.value)})}
                  />
                </div>
              </div>
              
              <div className="space-y-3">
                <div className="flex items-center justify-between p-3 border rounded-lg">
                  <div>
                    <div className="font-medium">Auto-confirm Orders</div>
                    <div className="text-sm text-gray-600">Automatically confirm order placement</div>
                  </div>
                  <input 
                    type="checkbox" 
                    checked={tradingPrefs.autoConfirmOrders}
                    onChange={(e) => setTradingPrefs({...tradingPrefs, autoConfirmOrders: e.target.checked})}
                    className="w-4 h-4"
                  />
                </div>
                
                <div className="flex items-center justify-between p-3 border rounded-lg">
                  <div>
                    <div className="font-medium">Show Advanced Features</div>
                    <div className="text-sm text-gray-600">Display advanced trading tools</div>
                  </div>
                  <input 
                    type="checkbox" 
                    checked={tradingPrefs.showAdvancedFeatures}
                    onChange={(e) => setTradingPrefs({...tradingPrefs, showAdvancedFeatures: e.target.checked})}
                    className="w-4 h-4"
                  />
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="api" className="space-y-4">
          <Card>
            <CardHeader>
              <div className="flex justify-between items-center">
                <CardTitle className="flex items-center gap-2">
                  <Key className="w-5 h-5" />
                  API Keys
                </CardTitle>
                <Button onClick={handleGenerateApiKey}>
                  <Plus className="w-4 h-4 mr-2" />
                  Generate New Key
                </Button>
              </div>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                {apiKeys.map((apiKey) => (
                  <div key={apiKey.id} className="p-4 border rounded-lg">
                    <div className="flex items-center justify-between mb-3">
                      <div>
                        <div className="font-medium">{apiKey.name}</div>
                        <div className="text-sm text-gray-600">
                          Created {apiKey.created} • Last used {apiKey.lastUsed}
                        </div>
                      </div>
                      <div className="flex items-center gap-2">
                        <Badge className={apiKey.status === 'active' ? 'bg-green-100 text-green-800' : 'bg-gray-100 text-gray-800'}>
                          {apiKey.status}
                        </Badge>
                        <Button size="sm" variant="outline" onClick={() => setShowApiKey(showApiKey === apiKey.id ? null : apiKey.id)}>
                          {showApiKey === apiKey.id ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
                        </Button>
                      </div>
                    </div>
                    
                    <div className="flex items-center gap-2 mb-3">
                      <Input 
                        value={showApiKey === apiKey.id ? apiKey.key : '••••••••••••••••••••'}
                        readOnly
                        className="font-mono text-sm"
                      />
                      <Button size="sm" variant="outline" onClick={() => copyToClipboard(apiKey.key)}>
                        <Copy className="w-4 h-4" />
                      </Button>
                    </div>
                    
                    <div className="flex items-center justify-between">
                      <div className="flex gap-2">
                        {apiKey.permissions.map(permission => (
                          <Badge key={permission} variant="outline">
                            {permission}
                          </Badge>
                        ))}
                      </div>
                      <div className="flex gap-2">
                        <Button size="sm" variant="outline">
                          <RefreshCw className="w-4 h-4" />
                        </Button>
                        <Button size="sm" variant="outline">
                          <Trash2 className="w-4 h-4" />
                        </Button>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="notifications" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Bell className="w-5 h-5" />
                Notification Preferences
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="space-y-3">
                <div className="flex items-center justify-between p-3 border rounded-lg">
                  <div>
                    <div className="font-medium flex items-center gap-2">
                      <Mail className="w-4 h-4" />
                      Email Notifications
                    </div>
                    <div className="text-sm text-gray-600">Receive notifications via email</div>
                  </div>
                  <input 
                    type="checkbox" 
                    checked={profile.emailNotifications}
                    onChange={(e) => setProfile({...profile, emailNotifications: e.target.checked})}
                    className="w-4 h-4"
                  />
                </div>
                
                <div className="flex items-center justify-between p-3 border rounded-lg">
                  <div>
                    <div className="font-medium flex items-center gap-2">
                      <Smartphone className="w-4 h-4" />
                      SMS Notifications
                    </div>
                    <div className="text-sm text-gray-600">Receive notifications via SMS</div>
                  </div>
                  <input 
                    type="checkbox" 
                    checked={profile.smsNotifications}
                    onChange={(e) => setProfile({...profile, smsNotifications: e.target.checked})}
                    className="w-4 h-4"
                  />
                </div>
                
                <div className="flex items-center justify-between p-3 border rounded-lg">
                  <div>
                    <div className="font-medium flex items-center gap-2">
                      <Bell className="w-4 h-4" />
                      Push Notifications
                    </div>
                    <div className="text-sm text-gray-600">Receive browser push notifications</div>
                  </div>
                  <input 
                    type="checkbox" 
                    checked={profile.pushNotifications}
                    onChange={(e) => setProfile({...profile, pushNotifications: e.target.checked})}
                    className="w-4 h-4"
                  />
                </div>
              </div>
              
              <div className="pt-4 border-t">
                <h3 className="font-medium mb-3">Notification Types</h3>
                <div className="grid gap-3 md:grid-cols-2">
                  <div className="space-y-2">
                    <label className="flex items-center gap-2">
                      <input type="checkbox" defaultChecked className="w-4 h-4" />
                      <span className="text-sm">Trade Executions</span>
                    </label>
                    <label className="flex items-center gap-2">
                      <input type="checkbox" defaultChecked className="w-4 h-4" />
                      <span className="text-sm">Price Alerts</span>
                    </label>
                    <label className="flex items-center gap-2">
                      <input type="checkbox" defaultChecked className="w-4 h-4" />
                      <span className="text-sm">Risk Alerts</span>
                    </label>
                  </div>
                  <div className="space-y-2">
                    <label className="flex items-center gap-2">
                      <input type="checkbox" defaultChecked className="w-4 h-4" />
                      <span className="text-sm">Market Updates</span>
                    </label>
                    <label className="flex items-center gap-2">
                      <input type="checkbox" className="w-4 h-4" />
                      <span className="text-sm">System Maintenance</span>
                    </label>
                    <label className="flex items-center gap-2">
                      <input type="checkbox" className="w-4 h-4" />
                      <span className="text-sm">Account Security</span>
                    </label>
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
} 