"use client";

import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { IranAssetBacktest } from "./_components/IranAssetBacktest";
import { FlaskConical } from "lucide-react";

export default function BacktestingPage() {
  return (
    <div className="min-h-screen bg-[#0a0b0f] text-white" dir="rtl">
      <div className="max-w-5xl mx-auto px-4 py-6 space-y-6">

        <div className="flex items-center gap-3">
          <div className="p-2 rounded-lg bg-purple-400/10 border border-purple-400/20">
            <FlaskConical className="w-5 h-5 text-purple-400" />
          </div>
          <div>
            <h1 className="text-xl font-bold">بک‌تستینگ</h1>
            <p className="text-xs text-muted-foreground mt-0.5">
              آزمون استراتژی بر اساس داده‌های تاریخی
            </p>
          </div>
        </div>

        <Tabs defaultValue="iran">
          <TabsList className="bg-white/5 border border-white/10 rounded-xl p-1 gap-0.5">
            <TabsTrigger value="iran" className="text-xs px-3 py-1.5 rounded-lg data-[state=active]:bg-white/10 data-[state=active]:text-white text-muted-foreground">
              🇮🇷 دارایی‌های ایرانی
            </TabsTrigger>
            <TabsTrigger value="global" className="text-xs px-3 py-1.5 rounded-lg data-[state=active]:bg-white/10 data-[state=active]:text-white text-muted-foreground">
              🌍 بازارهای جهانی
            </TabsTrigger>
          </TabsList>

          <TabsContent value="iran" className="mt-4">
            <IranAssetBacktest />
          </TabsContent>

          <TabsContent value="global" className="mt-4">
            <div className="flex items-center justify-center py-20 text-muted-foreground text-sm">
              بک‌تست بازارهای جهانی — به زودی
            </div>
          </TabsContent>
        </Tabs>

      </div>
    </div>
  );
}
