import type { Components, Theme } from '@mui/material';

export const MuiFormHelperText = (theme: Theme): Components<Theme>['MuiFormHelperText'] => {
  const {
    palette: { text },
    typography: { body2 },
  } = theme;

  return {
    defaultProps: {},
    styleOverrides: {
      root: {
        margin: 0,
        ...body2,
        color: text.disabled,
      },
    },
  };
};
