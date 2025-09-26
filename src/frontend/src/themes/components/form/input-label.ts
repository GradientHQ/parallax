import type { Components, InputLabelOwnProps, Theme } from '@mui/material';

export const MuiInputLabel = (theme: Theme): Components<Theme>['MuiInputLabel'] => {
  const {
    palette: { text },
    typography: { body2 },
  } = theme;
  return {
    defaultProps: {
      shrink: true,
      variant: 'outlined',
    },
    styleOverrides: {
      root: {
        position: 'relative',
        inset: 'unset',
        top: 'unset',
        left: 'unset',
        right: 'unset',
        bottom: 'unset',
        transform: 'none',
        transformOrigin: 'unset',
        maxWidth: 'unset',
        ...body2,
        color: text.secondary,
        backgroundColor: 'transparent',
      },
    },
  };
};
