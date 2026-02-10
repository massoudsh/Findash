import NextAuth from "next-auth";
import CredentialsProvider from "next-auth/providers/credentials";
import type { NextAuthOptions } from "next-auth";

const authOptions: NextAuthOptions = {
  providers: [
    CredentialsProvider({
      name: "Credentials",
      credentials: {
        email: { label: "Email", type: "email", placeholder: "jsmith@example.com" },
        password: { label: "Password", type: "password" },
      },
      async authorize(credentials) {
        // TODO: Replace with real backend call
        if (
          credentials?.email === "demo@demo.com" &&
          credentials?.password === "password"
        ) {
          return { id: "1", name: "Demo User", email: "demo@demo.com" };
        }
        return null;
      },
    }),
  ],
  session: {
    strategy: "jwt",
  },
  pages: {
    signIn: "/auth/signin",
    // signOut, error, verifyRequest, newUser can be added as needed
  },
};

const handler = NextAuth(authOptions);

export { handler as GET, handler as POST }; 