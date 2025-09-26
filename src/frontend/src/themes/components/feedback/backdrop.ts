import type { Components, Theme } from '@mui/material';

export const MuiBackdrop = (theme: Theme): Components<Theme>['MuiBackdrop'] => {
  return {
    styleOverrides: {
      root: {
        backgroundColor: 'rgba(25, 24, 24, 0.5)',
        variants: [
          {
            props: { invisible: true },
            style: { backgroundColor: 'transparent' },
          },
        ],
      },
    },
  };
};
