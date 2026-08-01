/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,ts,jsx,tsx}'],
  darkMode: 'class',
  theme: {
    extend: {
      fontFamily: {
        sans: ['Geist Variable', 'Inter', '-apple-system', 'BlinkMacSystemFont', 'Segoe UI', 'system-ui', 'sans-serif'],
        mono: ['Geist Mono Variable', 'ui-monospace', 'SFMono-Regular', 'Menlo', 'Monaco', 'Consolas', 'monospace'],
      },
      colors: {
        // Indigo → violet accent. Compressed range vs. stock Tailwind indigo so
        // 600/700 stay legible as text/icons on a near-black background instead
        // of going muddy — most usages sit directly on `canvas`/`surface`, not white.
        brand: {
          50:  '#f2f1ff',
          100: '#e5e3fe',
          200: '#cbc7fd',
          300: '#ada6fb',
          400: '#9089f8',
          500: '#7c6ef5',
          600: '#6c5ce9',
          700: '#5d4bd6',
          800: '#4c3bb0',
          900: '#3d2f8c',
        },
        // Page background + card surfaces for the dark theme.
        canvas: '#0a0a0f',
        surface: {
          DEFAULT: '#121218',
          raised: '#17171f',
          hover: '#1c1c26',
        },
        line: {
          DEFAULT: 'rgba(255,255,255,0.08)',
          strong: 'rgba(255,255,255,0.14)',
        },
      },
      boxShadow: {
        card: '0 1px 2px 0 rgb(0 0 0 / 0.4), 0 0 0 1px rgb(255 255 255 / 0.06)',
        'card-hover': '0 8px 24px -4px rgb(0 0 0 / 0.5), 0 0 0 1px rgb(255 255 255 / 0.1)',
        glow: '0 0 0 1px rgb(124 110 245 / 0.4), 0 0 24px -4px rgb(124 110 245 / 0.5)',
        panel: '0 4px 32px -8px rgb(0 0 0 / 0.6)',
      },
      backgroundImage: {
        'gradient-brand': 'linear-gradient(135deg, #7c6ef5 0%, #a78bfa 100%)',
        'gradient-radial-glow': 'radial-gradient(circle at 50% 0%, rgb(124 110 245 / 0.16), transparent 60%)',
      },
    },
  },
  plugins: [],
}
