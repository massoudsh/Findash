"use client";

import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { PortfolioAssetsSection } from "./_components/PortfolioAssetsSection";
import { PieChart } from "lucide-react";

export default function PortfolioPage() {
  return (
    <div className="min-h-screen bg-[#0a0b0f] text-white" dir="rtl">
      <div className="max-w-5xl mx-auto px-4 py-6 space-y-6">

        {/* Header */}
        <div className="flex items-center gap-3">
          <div className="p-2 rounded-lg bg-orange-400/10 border border-orange-400/20">
            <PieChart className="w-5 h-5 text-orange-400" />
          </div>
          <div>
            <h1 className="text-xl font-bold">پورتفولیو</h1>
            <p className="text-xs text-muted-foreground mt-0.5">
              مدیریت و تحلیل دارایی‌های شما
            </p>
          </div>
        </div>

        {/* Tabs */}
        <Tabs defaultValue="assets">
          <TabsList className="bg-white/5 border border-white/10 rounded-xl p-1 gap-0.5">
            <TabsTrigger
              value="assets"
              className="text-xs px-3 py-1.5 rounded-lg data-[state=active]:bg-white/10 data-[state=active]:text-white text-muted-foreground"
            >
              🏦 دارایی‌های فیزیکی
            </TabsTrigger>
            <TabsTrigger
              value="stocks"
              className="text-xs px-3 py-1.5 rounded-lg data-[state=active]:bg-white/10 data-[state=active]:text-white text-muted-foreground"
            >
              📈 سهام و اوراق
            </TabsTrigger>
            <TabsTrigger
              value="analysis"
              className="text-xs px-3 py-1.5 rounded-lg data-[state=active]:bg-white/10 data-[state=active]:text-white text-muted-foreground"
            >
              📊 آنالیز
            </TabsTrigger>
          </TabsList>

          <TabsContent value="assets" className="mt-4">
            <PortfolioAssetsSection />
          </TabsContent>

          <TabsContent value="stocks" className="mt-4">
            <div className="flex items-center justify-center py-20 text-muted-foreground text-sm">
              ماژول سهام — به زودی
            </div>
          </TabsContent>

          <TabsContent value="analysis" className="mt-4">
            <div className="flex items-center justify-center py-20 text-muted-foreground text-sm">
              آنالیز پورتفولیو — به زودی
            </div>
          </TabsContent>
        </Tabs>

      </div>
    </div>
  );
}
