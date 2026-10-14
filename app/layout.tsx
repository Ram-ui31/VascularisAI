import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import "./globals.css";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  metadataBase: new URL("https://vascularis-ai.mistymacaw1.chatgpt.site"),
  title: "Vascularis — Your heart, in focus",
  description: "Explore your clinical measurements and coronary insights with interactive heart anatomy.",
  openGraph: {
    title: "Vascularis AI",
    description: "Cardiac risk, made visible.",
    images: [{ url: "/og.png", width: 1733, height: 908, alt: "Vascularis AI cardiac digital twin" }],
  },
  twitter: {
    card: "summary_large_image",
    title: "Vascularis AI",
    description: "Cardiac risk, made visible.",
    images: ["/og.png"],
  },
  icons: {
    icon: "/favicon.svg",
    shortcut: "/favicon.svg",
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body
        className={`${geistSans.variable} ${geistMono.variable} antialiased`}
      >
        {children}
      </body>
    </html>
  );
}
