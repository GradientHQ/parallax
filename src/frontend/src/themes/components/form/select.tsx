import type { Components, SelectProps, Theme } from '@mui/material';
import { IconButton } from '@mui/material';
import { IconChevronDown } from '@tabler/icons-react';
import { INPUT_SIZE_PADDING_REM_MAP } from '../constants';

const SIZES: NonNullable<SelectProps['size']>[] = ['small', 'medium'];

export const MuiSelect = (theme: Theme): Components<Theme>['MuiSelect'] => {
  const { palette } = theme;
  return {
    defaultProps: {
      size: 'small',
      variant: 'outlined',
      IconComponent: ({ ownerState, ...props }: { ownerState: SelectProps }) => (
        <IconButton {...(ownerState as any)} {...props}>
          <IconChevronDown />
        </IconButton>
      ),
    },
    styleOverrides: {
      root: {},
      select: {
        // position: 'absolute',
        // inset: 0,
        // top: 0,
        // left: 0,
        // right: 0,
        // bottom: 0,
        // display: 'flex'
      },
      icon: {
        top: '50%',
        transform: 'translateY(-50%)',
        color: palette.text.disabled,

        variants: [
          {
            props: ({ ownerState }: { ownerState: SelectProps }) => !!ownerState.open,
            style: {
              transform: 'translateY(-50%) rotate(180deg)',
            },
          },
          ...SIZES.map((size) => ({
            props: { size },
            style: {
              right: `${INPUT_SIZE_PADDING_REM_MAP[size]}rem`,
            },
          })),
        ],
      },
    },
  };
};
