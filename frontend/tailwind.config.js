/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  darkMode: 'class', // <--- CRITICAL FOR LIGHT/DARK MODE
  theme: {
    extend: {
      fontFamily: {
        sans: ['Inter', 'system-ui', 'sans-serif'], // Cleaner font stack
      },
      colors: {
        // Professional Brand Color (Indigo/Slate mix)
        brand: {
          50: '#f0f9ff',
          100: '#e0f2fe',
          500: '#0ea5e9', // Primary Action
          600: '#0284c7',
          700: '#0369a1',
          900: '#0c4a6e',
        },
        // Keep board colors for game component
        board: "#1e40af",
        slot: "#ffffff",
      }
    },
  },
  plugins: [],
}