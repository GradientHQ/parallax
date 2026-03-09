import type { Components, InputBaseProps, Theme } from '@mui/material';
import { INPUT_RADIUS, INPUT_SIZE_PADDING_REM_MAP, INPUT_SIZE_REM_MAP } from '../constants';

const SIZES: NonNullable<InputBaseProps['size']>[] = ['small', 'medium'];

export const MuiInputBase = (theme: Theme): Components<Theme>['MuiInputBase'] => {
  const {
    palette: { grey, text },
    spacing,
    typography: { body2 },
  } = theme;

  const sizeVariants = SIZES.map((size) => [
    {
      props: { size },
      style: {
        height: `${INPUT_SIZE_REM_MAP[size]}rem`,
        paddingInline: `${INPUT_SIZE_PADDING_REM_MAP[size]}rem`,
      },
    },
    {
      props: { size, multiline: true },
      style: {
        height: 'auto',
        paddingBlock: `${INPUT_SIZE_PADDING_REM_MAP[size]}rem`,
      },
    },
  ]).flat(1);

  return {
    defaultProps: {
      size: 'small',
    },
    styleOverrides: {
      root: {
        padding: 0,
        borderRadius: `${INPUT_RADIUS}rem !important`,
        gap: spacing(1),
        variants: [...sizeVariants],
      },

      input: {
        ...body2,
        color: text.primary,
        padding: 0,
        textAlign: 'start',

        '&:placeholder': {
          color: grey[400],
        },

        variants: [
          {
            props: { type: 'number' },
            style: {
              textAlign: 'end',
            },
          },
        ],
      },
    },
  };
};
