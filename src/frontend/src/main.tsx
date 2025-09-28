import './global.css';

import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import { HashRouter } from 'react-router-dom';

import { ThemeProvider } from './themes';
import { App } from './App';
import { CssBaseline } from '@mui/material';

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <HashRouter>
      <ThemeProvider>
        <CssBaseline />
        <App />
      </ThemeProvider>
    </HashRouter>
  </StrictMode>,
);
