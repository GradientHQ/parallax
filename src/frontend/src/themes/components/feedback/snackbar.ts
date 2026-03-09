import type { Components, Theme } from '@mui/material';

export const MuiSnackbarContent = (theme: Theme): Components<Theme>['MuiSnackbarContent'] => {
  const { palette, typography } = theme;

  return {
    defaultProps: {
      variant: 'overlay',
    },
    styleOverrides: {
      root: {
        ...typography.subtitle1,
        color: palette.text.primary,
        backgroundColor: palette.background.default,
        padding: 0,
      },
      message: {
        padding: 0,
      },
    },
  };
};

export const MuiSnackbar = (theme: Theme): Components<Theme>['MuiSnackbar'] => {
  return {
    defaultProps: {
      anchorOrigin: {
        vertical: 'bottom',
        horizontal: 'right',
      },
    },
    styleOverrides: {
      root: {},
    },
  };
};
