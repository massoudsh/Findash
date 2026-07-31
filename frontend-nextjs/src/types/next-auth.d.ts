import "next-auth";

// route.ts (`callbacks.session`) واقعاً `session.user.id`/`session.accessToken` را ست می‌کند؛
// این فایل فقط تایپ رسمی next-auth را با شکل واقعی session هماهنگ می‌کند.
declare module "next-auth" {
  interface Session {
    accessToken?: string;
    refreshToken?: string;
    user: {
      id?: string;
      name?: string | null;
      email?: string | null;
      image?: string | null;
    };
  }
}
