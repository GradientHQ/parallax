import type { Components, Theme } from '@mui/material';
import type { LinearProgressProps } from '@mui/material';

declare module '@mui/material/LinearProgress' {
  interface LinearProgressPropsColorOverrides {
    brand: true;
  }
}

const COLORS: NonNullable<LinearProgressProps['color']>[] = [
  'primary',
  'secondary',
  'brand',
  'info',
  'error',
  'success',
  'warning',
];

export const MuiLinearProgress = (theme: Theme): Components<Theme>['MuiLinearProgress'] => {
  const { palette } = theme;
  return {
    styleOverrides: {
      root: {
        height: '0.375rem',
        borderRadius: '0.1875rem',
        variants: [
          ...COLORS.map((color) => ({
            props: { color },
            style: { backgroundColor: palette.grey[300] },
          })),
        ],
      },
      bar1: {
        variants: [
          ...COLORS.map((color) => ({
            props: { color },
            style: { backgroundColor: color === 'inherit' ? 'currentColor' : palette[color].main },
          })),
        ],
      },
    },
  };
};
