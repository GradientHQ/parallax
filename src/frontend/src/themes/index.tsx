'use client';

import type { FC, PropsWithChildren } from 'react';

import type { Theme } from '@mui/material';
import { createTheme, THEME_ID, ThemeProvider } from '@mui/material';

import { LocalizationProvider } from '@mui/x-date-pickers/LocalizationProvider';
import { AdapterDayjs } from '@mui/x-date-pickers/AdapterDayjs';

import { palette } from './palette';
import { HTML_FONT_SIZE, typography } from './typography';
import { overlays } from './shadows';
import * as themeComponents from './components';
import { SnackbarProvider } from './components';

const materialThemeLight = createTheme({
  palette,
  typography,
  spacing: (factor: number) => `${(factor * 8) / HTML_FONT_SIZE}rem`,
  overlays,
});

materialThemeLight.components = materialThemeLight.components || {};

(
  Object.entries(themeComponents) as [
    keyof NonNullable<Theme['components']>,
    (theme: Theme) => NonNullable<Theme['components']>[keyof NonNullable<Theme['components']>],
  ][]
).forEach(
  <K extends keyof NonNullable<Theme['components']>>([compName, generate]: [
    K,
    (theme: Theme) => NonNullable<Theme['components']>[K],
  ]) => {
    if (compName.startsWith('Mui') && typeof generate === 'function') {
      materialThemeLight.components![compName] = generate(materialThemeLight);
    }
  },
);

const Provider: FC<PropsWithChildren> = ({ children }) => {
  return (
    <ThemeProvider theme={{ [THEME_ID]: materialThemeLight }}>
      <SnackbarProvider>
        <LocalizationProvider dateAdapter={AdapterDayjs} localeText={{ okButtonLabel: 'Apply' }}>
          {children}
        </LocalizationProvider>
      </SnackbarProvider>
    </ThemeProvider>
  );
};

export { Provider as ThemeProvider };
