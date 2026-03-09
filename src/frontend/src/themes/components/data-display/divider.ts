import type { Components, Theme } from '@mui/material';
import { INPUT_ICON_SIZE_REM_MAP } from '../constants';

export const MuiDivider = (theme: Theme): Components<Theme>['MuiDivider'] => {
  const { spacing } = theme;

  return {
    styleOverrides: {
      root: {
        variants: [
          {
            props: { variant: 'inset' },
            style: {
              margin: 0,
              marginInlineStart: `${INPUT_ICON_SIZE_REM_MAP.small}rem`,
            },
          },
        ],
      },
    },
  };
};
