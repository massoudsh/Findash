"use client";
import { useSession, signOut } from "next-auth/react";

export function UserMenu() {
  const { data: session, status } = useSession();

  if (status === "loading") {
    return <div className="px-4">در حال بارگذاری...</div>;
  }

  if (session) {
    return (
      <div className="flex items-center gap-4 px-4">
        <span className="text-sm text-gray-700">{session.user?.email}</span>
        <button
          onClick={() => signOut({ callbackUrl: "/auth/signin" })}
          className="bg-red-500 text-white px-4 py-1 rounded-md font-semibold hover:bg-red-600 transition"
        >
          خروج
        </button>
      </div>
    );
  }

  // When not signed in, show a generic avatar
  return (
    <div className="flex items-center gap-4 px-4">
      <span className="inline-flex items-center justify-center w-8 h-8 rounded-full bg-gray-200 text-gray-500 text-xl">
        <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor" className="w-6 h-6">
          <path strokeLinecap="round" strokeLinejoin="round" d="M15.75 7.5a3.75 3.75 0 11-7.5 0 3.75 3.75 0 017.5 0zM4.5 19.5a7.5 7.5 0 0115 0v.75a.75.75 0 01-.75.75h-13.5a.75.75 0 01-.75-.75V19.5z" />
        </svg>
      </span>
    </div>
  );
}

export default UserMenu; 