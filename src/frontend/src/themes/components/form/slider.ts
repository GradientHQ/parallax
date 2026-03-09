import type { Components, Palette, PaletteColor, SliderProps, Theme } from '@mui/material';

declare module '@mui/material/Slider' {
  interface SliderPropsColorOverrides {
    brand: true;
  }
}

const Sizes: readonly NonNullable<SliderProps['size']>[] = ['small', 'medium'];

// unit rem
const SizeMaps: Record<
  NonNullable<SliderProps['size']>,
  {
    thumb: number;
    track: number;
    rail: number;
    padding: number;
  }
> = {
  small: {
    thumb: 0.75,
    track: 0.3125,
    rail: 0.3125,
    padding: 0.125,
  },
  medium: {
    thumb: 1,
    track: 0.625,
    rail: 0.625,
    padding: 0.125,
  },
};

const COLORS: readonly NonNullable<SliderProps['color']>[] = [
  'brand',
  'primary',
  'secondary',
  'error',
  'info',
  'success',
  'warning',
];

export const MuiSlider = (theme: Theme): Components<Theme>['MuiSlider'] => {
  const colorVariants = COLORS.map((color) => {
    const paletteColor = theme.palette[color as keyof Palette] as PaletteColor;
    return Sizes.map((size) => {
      const sizeClassKey = `&.MuiSlider-size${size.charAt(0).toUpperCase() + size.slice(1)}`;
      return {
        props: { color, size },
        style: {
          color: paletteColor.main,
          '& .MuiSlider-thumb': {
            backgroundColor: paletteColor.main,
            '&.Mui-disabled': {
              backgroundColor: theme.palette.grey[250],
              '&::before': {
                boxShadow: 'none',
              },
            },

            '&:hover, &.Mui-focusVisible, &.Mui-active': {
              boxShadow: '0px 0px 0px 0.25rem rgba(5, 170, 108, 0.16)',
            },
          },

          // size
          [sizeClassKey]: {
            '.MuiSlider-thumb': {
              width: `${SizeMaps[size].thumb}rem`,
              height: `${SizeMaps[size].thumb}rem`,
            },
            '.MuiSlider-track': {
              height: `${SizeMaps[size].track - SizeMaps[size].padding * 2}rem`,
              marginLeft: `${SizeMaps[size].padding}rem`,
            },
            '.MuiSlider-rail': {
              height: `${SizeMaps[size].rail}rem`,
              padding: `0 ${SizeMaps[size].padding}rem`,
              backgroundColor: theme.palette.grey[250],
              opacity: 1,
            },
          },
        },
      };
    });
  }).flat(1);

  return {
    defaultProps: {
      size: 'medium',
      color: 'brand',
    },
    styleOverrides: {
      root: {
        paddingBottom: 0,
      },
      thumb: {
        borderWidth: '0.2rem',
        borderStyle: 'solid',
        borderColor: theme.palette.common.white,
      },
    },
    variants: colorVariants,
  };
};
