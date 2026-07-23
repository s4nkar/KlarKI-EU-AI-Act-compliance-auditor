/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,ts,jsx,tsx}'],
  theme: {
    // Neo-brutalism: flatten every corner except full circles (dots, avatars).
    borderRadius: {
      none: '0px',
      sm: '0px',
      DEFAULT: '0px',
      md: '0px',
      lg: '0px',
      xl: '0px',
      '2xl': '0px',
      '3xl': '0px',
      full: '9999px',
    },
    extend: {
      fontFamily: {
        sans: ['Inter', '-apple-system', 'BlinkMacSystemFont', 'Segoe UI', 'system-ui', 'sans-serif'],
      },
      colors: {
        // Repurposed as the primary coral/red CTA accent (was indigo).
        brand: {
          50:  '#fff1ef',
          100: '#ffe1dc',
          200: '#ffc3ba',
          300: '#ff9c8c',
          400: '#ff7a63',
          500: '#ff6b52',
          600: '#f2452f',
          700: '#d1301c',
          800: '#a02717',
          900: '#7a1f12',
        },
        cream: '#faf5e9',
        yellow: {
          400: '#ffdd55',
          500: '#f5c518',
        },
        lavender: '#c9bffa',
      },
      boxShadow: {
        // Hard offset shadows, no blur — the signature neo-brutalist "sticker" look.
        sm: '2px 2px 0 0 #000',
        DEFAULT: '3px 3px 0 0 #000',
        md: '4px 4px 0 0 #000',
        lg: '6px 6px 0 0 #000',
        xl: '8px 8px 0 0 #000',
        '2xl': '10px 10px 0 0 #000',
        card: '4px 4px 0 0 #000',
        'card-hover': '6px 6px 0 0 #000',
        nav: '0 3px 0 0 #000',
      },
      backgroundImage: {
        'gradient-brand': 'linear-gradient(135deg, #ff6b52 0%, #f2452f 100%)',
      },
    },
  },
  plugins: [],
}
