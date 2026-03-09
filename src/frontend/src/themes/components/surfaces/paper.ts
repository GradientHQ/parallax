import type { Components, Theme } from '@mui/material';

declare module '@mui/material/Paper' {
  interface PaperPropsVariantOverrides {
    overlay: true;
  }
}

export const MuiPaper = (theme: Theme): Components<Theme>['MuiPaper'] => {
  const { palette, overlays } = theme;

  return {
    defaultProps: {
      variant: 'outlined',
    },
    styleOverrides: {
      root: {
        variants: [
          {
            props: ({ ownerState }) => !ownerState.square,
            style: {
              borderRadius: '0.75rem',
            },
          },
          {
            props: { variant: 'outlined' },
            style: {
              border: `1px solid ${palette.divider}`,
            },
          },
          {
            props: { variant: 'overlay' },
            style: {
              border: `1px solid ${palette.divider}`,
              boxShadow: overlays.shadowMiddle,
            },
          },
        ],
      },
    },
  };
};
