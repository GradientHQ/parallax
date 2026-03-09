import GeistMonoTTF from '../../assets/fonts/GeistMono.ttf';
import InterTTF from '../../assets/fonts/Inter.ttf';
import FKGroteskNeueWoff2 from '../../assets/fonts/FKGroteskNeue.woff2';

import type { Components, Theme } from '@mui/material';

export const MuiCssBaseline = (theme: Theme): Components<Theme>['MuiCssBaseline'] => {
  return {
    styleOverrides: `
      @font-face {
        font-family: 'GeistMono';
        src: url(${GeistMonoTTF}) format('ttf');
      }
      @font-face {
        font-family: 'Inter';
        src: url(${InterTTF}) format('ttf');
      }
      @font-face {
        font-family: 'FK Grotesk Neue';
        src: url(${FKGroteskNeueWoff2}) format('woff2');
      }
    `,
  };
};
