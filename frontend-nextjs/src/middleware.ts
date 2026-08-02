import { withAuth } from "next-auth/middleware";
import { NextResponse } from "next/server";

// موقتاً باز: تا زمانی که سیستم تفکیک محتوای پرمیوم/رایگان پیاده‌سازی شود،
// همه‌ی صفحات (dashboard/portfolio/trading/analytics/settings) بدون ورود هم قابل مشاهده‌اند.
// برای بازگرداندن محدودیت لاگین، خط `authorized` را به `!!token` برگردانید.
export default withAuth(
  function middleware(req) {
    return NextResponse.next();
  },
  {
    callbacks: {
      authorized: () => true,
    },
    pages: {
      signIn: "/auth/signin",
    },
  }
);

export const config = {
  matcher: [
    "/dashboard/:path*",
    "/portfolio/:path*",
    "/trading/:path*",
    "/analytics/:path*",
    "/settings/:path*",
  ],
};
