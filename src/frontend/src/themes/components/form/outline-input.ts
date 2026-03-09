import type { Components, Theme } from '@mui/material';
import { outlinedInputClasses } from '@mui/material';

export const MuiOutlinedInput = (theme: Theme): Components<Theme>['MuiOutlinedInput'] => {
  const {
    palette: { grey, error, text, divider, background },
    typography: { body2 },
    overlays,
  } = theme;
  return {
    defaultProps: {
      size: 'small',
    },
    styleOverrides: {
      root: {
        backgroundColor: background.default,
        boxShadow: overlays.shadowInput,

        '&:hover': {
          backgroundColor: grey[50],
        },

        [`&.${outlinedInputClasses.focused}`]: {
          boxShadow: overlays.shadowInputActive,
        },

        [`&.${outlinedInputClasses.disabled}`]: {
          '&, &:hover': {
            backgroundColor: grey[200],
          },
        },

        [`&, &.${outlinedInputClasses.focused}, &.${outlinedInputClasses.disabled}`]: {
          '&, &:hover': {
            [`.${outlinedInputClasses.notchedOutline}`]: {
              borderWidth: 1,
              borderColor: divider,
            },
          },
        },

        [`&.${outlinedInputClasses.error}`]: {
          boxShadow: overlays.shadowInputError,
          '&, &:hover': {
            [`.${outlinedInputClasses.notchedOutline}`]: {
              borderWidth: 1,
              borderColor: error.main,
            },
          },
        },
      },

      input: {
        ...body2,
        color: text.primary,
        padding: 0,
      },

      notchedOutline: {
        top: 0,
        left: 0,
        right: 0,
        bottom: 0,

        border: `1px solid`,
        borderColor: divider,

        variants: [
          {
            props: { disabled: true },
            style: {
              borderColor: divider,
            },
          },
        ],

        [`legend`]: {
          display: 'none',
        },
      },
    },
  };
};
