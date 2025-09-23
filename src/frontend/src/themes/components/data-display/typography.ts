import type { Components, Theme } from '@mui/material';

declare module '@mui/material' {
  interface TypographyPropsVariantOverrides {
    pre: true;
  }
}

export const MuiTypography = (theme: Theme): Components<Theme>['MuiTypography'] => {
  return {
    defaultProps: {
      variant: 'inherit',
      color: 'inherit',
      variantMapping: {
        pre: 'pre',
      },
    },
    styleOverrides: {
      root: {
        flex: 'none',
        variants: [
          {
            props: { variant: 'pre' },
            style: {
              fontFamily: 'Geist Mono, monospace',
            },
          },
        ],
      },
    },
  };
};
