import js from '@eslint/js'
import globals from 'globals'
import react from 'eslint-plugin-react'
import reactHooks from 'eslint-plugin-react-hooks'
import reactRefresh from 'eslint-plugin-react-refresh'

// Flat config; the previously-imported `defineConfig` / `globalIgnores` come
// from ESLint v9 only — installed version is 8.57, so we just export the
// array directly.
export default [
  { ignores: ['dist'] },
  {
    files: ['**/*.{js,jsx}'],
    plugins: {
      react,
      'react-hooks': reactHooks,
      'react-refresh': reactRefresh,
    },
    languageOptions: {
      ecmaVersion: 2020,
      globals: globals.browser,
      parserOptions: {
        ecmaVersion: 'latest',
        ecmaFeatures: { jsx: true },
        sourceType: 'module',
      },
    },
    settings: {
      react: { version: '18' },
    },
    rules: {
      ...js.configs.recommended.rules,
      ...reactHooks.configs.recommended.rules,
      // `react/jsx-uses-vars` teaches no-unused-vars about JSX so that
      // `import { Foo }` followed by `<Foo />` is no longer flagged. Without
      // it, every Icon-as-prop pattern (`{ icon: Icon }` then `<Icon />`)
      // gets a false positive.
      'react/jsx-uses-vars': 'error',
      'react/jsx-uses-react': 'error',
      // Allow context files to export a Provider component PLUS a `useFoo`
      // hook + helpers from the same file — that's our convention.
      'react-refresh/only-export-components': [
        'warn',
        { allowConstantExport: true, allowExportNames: ['useDatabase', 'useTheme', 'useToast', 'useToastQueue', 'AllProviders', 'dbWrapper'] },
      ],
      'no-unused-vars': ['error', { varsIgnorePattern: '^[A-Z_]', argsIgnorePattern: '^_' }],
      // Surface accidental console.log creep, but allow .error / .warn for
      // genuine diagnostics (ErrorBoundary uses .error, tokenCleanup uses .warn).
      'no-console': ['warn', { allow: ['error', 'warn'] }],
    },
  },
]
