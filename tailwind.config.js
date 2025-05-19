// tailwind.config.js
module.exports = {
  darkMode: 'class', // This enables dark mode based on the presence of the 'dark' class
  content: [
    "./src/**/*.{js,jsx,ts,tsx}", // Or wherever your components are
  ],
  theme: {
    extend: {
      // You can extend colors here if needed
    },
  },
  plugins: [],
}