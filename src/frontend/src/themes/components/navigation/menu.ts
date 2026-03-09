import type { Components, Theme } from '@mui/material';
import { dividerClasses, listItemIconClasses } from '@mui/material';
import { INPUT_ICON_SIZE_REM_MAP } from '../constants';

export const MuiMenu = (theme: Theme): Components<Theme>['MuiMenu'] => {
  const { spacing } = theme;

  return {
    defaultProps: {
      slotProps: {
        paper: {
          variant: 'overlay',
        },
        backdrop: {
          invisible: true,
        },
      },
    },
    styleOverrides: {
      root: {
        [`& .${dividerClasses.root}`]: {
          marginBlock: spacing(1.5),
        },
      },
      paper: {
        padding: spacing(1),
      },
    },
  };
};

export const MuiMenuList = (theme: Theme): Components<Theme>['MuiMenuList'] => {
  const { spacing } = theme;

  return {
    defaultProps: {},
    styleOverrides: {},
  };
};

export const MuiMenuItem = (theme: Theme): Components<Theme>['MuiMenuItem'] => {
  const { palette, spacing } = theme;

  return {
    defaultProps: {},
    styleOverrides: {
      root: {
        padding: spacing(1),
        gap: spacing(0.75),
        '&:hover': {
          backgroundColor: palette.grey[200],
        },
        [`& + .${dividerClasses.root}`]: {
          margin: 0,
          marginBlock: spacing(1.5),
        },
        [`& .${dividerClasses.inset}`]: {
          margin: 0,
          marginInline: `${INPUT_ICON_SIZE_REM_MAP.small}rem`,
        },
        [`& .${listItemIconClasses.root}`]: {
          minWidth: `${INPUT_ICON_SIZE_REM_MAP.small}rem`,
        },
      },
    },
  };
};
