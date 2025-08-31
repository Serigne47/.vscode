import type { Metadata, Viewport } from 'next'
import { Inter } from 'next/font/google'
import { Toaster } from 'react-hot-toast'
import './globals.css'

// Configuration de la police Inter avec optimisations
const inter = Inter({
  subsets: ['latin'],
  display: 'swap',
  variable: '--font-inter',
  preload: true,
  fallback: ['system-ui', '-apple-system', 'sans-serif'],
})

// Métadonnées SEO optimisées
export const metadata: Metadata = {
  title: {
    default: 'Tender Analysis System',
    template: '%s | Tender Analysis',
  },
  description: 'Advanced AI-powered tender analysis system with 6 specialized agents for comprehensive document analysis',
  keywords: [
    'tender analysis',
    'RFP analysis',
    'procurement',
    'logistics',
    'AI analysis',
    'document processing',
  ],
  authors: [{ name: 'Tender Analysis Team' }],
  creator: 'Tender Analysis System',
  publisher: 'Your Company',
  
  // Open Graph
  openGraph: {
    type: 'website',
    locale: 'en_US',
    url: 'https://your-domain.com',
    siteName: 'Tender Analysis System',
    title: 'Tender Analysis System',
    description: 'AI-powered tender document analysis',
    images: [
      {
        url: '/og-image.png',
        width: 1200,
        height: 630,
        alt: 'Tender Analysis System',
      },
    ],
  },
  
  // Twitter Card
  twitter: {
    card: 'summary_large_image',
    title: 'Tender Analysis System',
    description: 'AI-powered tender document analysis',
    images: ['/twitter-image.png'],
  },
  
  // Icons
  icons: {
    icon: [
      { url: '/favicon.ico' },
      { url: '/favicon-16x16.png', sizes: '16x16', type: 'image/png' },
      { url: '/favicon-32x32.png', sizes: '32x32', type: 'image/png' },
    ],
    apple: [
      { url: '/apple-touch-icon.png' },
    ],
  },
  
  // Manifest
  manifest: '/manifest.json',
  
  // Verification
  verification: {
    google: 'your-google-verification-code',
    // yandex: 'your-yandex-verification-code',
    // bing: 'your-bing-verification-code',
  },
  
  // Robots
  robots: {
    index: true,
    follow: true,
    googleBot: {
      index: true,
      follow: true,
      'max-video-preview': -1,
      'max-image-preview': 'large',
      'max-snippet': -1,
    },
  },
  
  // App specific
  applicationName: 'Tender Analysis System',
  generator: 'Next.js',
}

// Configuration du viewport
export const viewport: Viewport = {
  width: 'device-width',
  initialScale: 1,
  maximumScale: 5,
  userScalable: true,
  themeColor: [
    { media: '(prefers-color-scheme: light)', color: '#ffffff' },
    { media: '(prefers-color-scheme: dark)', color: '#0f172a' },
  ],
}

// Providers wrapper component
function Providers({ children }: { children: React.ReactNode }) {
  return (
    <>
      {/* Toast notifications */}
      <Toaster
        position="top-right"
        reverseOrder={false}
        gutter={8}
        containerClassName=""
        containerStyle={{}}
        toastOptions={{
          // Default options
          duration: 4000,
          style: {
            background: 'hsl(var(--background))',
            color: 'hsl(var(--foreground))',
            border: '1px solid hsl(var(--border))',
            borderRadius: 'var(--radius)',
            fontSize: '14px',
          },
          // Success options
          success: {
            duration: 3000,
            iconTheme: {
              primary: '#10b981',
              secondary: '#ffffff',
            },
            style: {
              background: '#10b981',
              color: '#ffffff',
            },
          },
          // Error options
          error: {
            duration: 5000,
            iconTheme: {
              primary: '#ef4444',
              secondary: '#ffffff',
            },
            style: {
              background: '#ef4444',
              color: '#ffffff',
            },
          },
          // Loading options
          loading: {
            duration: Infinity,
            style: {
              background: 'hsl(var(--muted))',
              color: 'hsl(var(--muted-foreground))',
            },
          },
        }}
      />
      
      {/* Future providers can be added here */}
      {/* <ThemeProvider> */}
      {/* <AuthProvider> */}
      {/* <ReactQueryProvider> */}
      
      {children}
    </>
  )
}

// Layout racine
export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html 
      lang="en" 
      className={inter.variable}
      suppressHydrationWarning
    >
      <head>
        {/* Preconnect to external domains */}
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link rel="preconnect" href="https://fonts.gstatic.com" crossOrigin="anonymous" />
        
        {/* DNS Prefetch for API */}
        <link rel="dns-prefetch" href={process.env.NEXT_PUBLIC_API_URL || 'http://127.0.0.1:8000'} />
        
        {/* PWA meta tags */}
        <meta name="apple-mobile-web-app-capable" content="yes" />
        <meta name="apple-mobile-web-app-status-bar-style" content="default" />
        <meta name="apple-mobile-web-app-title" content="Tender Analysis" />
        <meta name="mobile-web-app-capable" content="yes" />
        
        {/* Security headers */}
        <meta httpEquiv="X-UA-Compatible" content="IE=edge" />
        <meta name="format-detection" content="telephone=no" />
        
        {/* Performance hints */}
        <meta name="renderer" content="webkit" />
        <meta name="x-dns-prefetch-control" content="on" />
      </head>
      
      <body 
        className={`${inter.className} min-h-screen bg-background font-sans antialiased`}
        suppressHydrationWarning
      >
        {/* Skip to content for accessibility */}
        <a 
          href="#main-content" 
          className="sr-only focus:not-sr-only focus:absolute focus:top-4 focus:left-4 z-50 bg-background text-foreground px-4 py-2 rounded-md"
        >
          Skip to main content
        </a>
        
        <Providers>
          {/* Main application */}
          <div className="relative flex min-h-screen flex-col">
            {/* Background pattern (optional) */}
            <div className="fixed inset-0 -z-10 h-full w-full bg-white dark:bg-gray-950">
              <div className="absolute h-full w-full bg-[radial-gradient(#e5e7eb_1px,transparent_1px)] dark:bg-[radial-gradient(#1f2937_1px,transparent_1px)] [background-size:20px_20px] [mask-image:radial-gradient(ellipse_60%_50%_at_50%_0%,#000_70%,transparent_100%)]" />
            </div>
            
            {/* Main content */}
            <main id="main-content" className="flex-1">
              {children}
            </main>
          </div>
        </Providers>
        
        {/* Scripts for performance monitoring (optional) */}
        {process.env.NODE_ENV === 'production' && (
          <script
            dangerouslySetInnerHTML={{
              __html: `
                // Performance monitoring
                if (typeof window !== 'undefined' && window.performance) {
                  window.addEventListener('load', function() {
                    const perfData = window.performance.timing;
                    const pageLoadTime = perfData.loadEventEnd - perfData.navigationStart;
                    console.log('Page load time:', pageLoadTime + 'ms');
                  });
                }
              `,
            }}
          />
        )}
      </body>
    </html>
  )
}