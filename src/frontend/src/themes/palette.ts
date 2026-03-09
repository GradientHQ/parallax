import type { Color, PaletteColorOptions, PaletteOptions } from '@mui/material';

declare module '@mui/material' {
  interface Color {
    150: string;
    250: string;
    1000: string;
  }
  interface PaletteColor {
    darker: string;
    lighter: string;
    chip: string;
    chipText: string;
  }
  interface SimplePaletteColorOptions {
    darker: string;
    lighter: string;
    chip: string;
    chipText: string;
  }

  interface Palette {
    brand: PaletteColor;
  }
  interface PaletteOptions {
    brand?: PaletteColorOptions;
  }

  interface TypeBackground {
    /** Use for some small area. */
    area: string;
  }
}

declare module '@mui/material/Chip' {
  interface ChipPropsColorOverrides {
    brand: true;
  }
}

const white = '#ffffff';
const black = '#000000';

const brand: Partial<Color> = {
  500: '#05aa6c',
  100: '#cdeee4',
  50: '#dff0e7',
};
const grey: Partial<Color> = {
  50: '#FAFAFA',
  100: '#F7F7F7',
  //150: '#f1f3eb',
  200: '#F1F1F1',
  250: '#e4e4e4',
  300: '#D6D6D6',
  400: '#BBBBBB',
  500: '#A0A0A0',
  600: '#858585',
  700: '#6A6969',
  800: '#4F4E4E',
  900: '#343333',
  1000: '#191818',
};
const red: Partial<Color> = {
  100: '#f5e5dd',
  200: '#efbcb3',
  400: '#f06a64',
  500: '#d92d20',
};
const orange: Partial<Color> = {
  100: '#f5e9cf',
  200: '#e99f55',
  500: '#ec7804',
};
const green: Partial<Color> = {
  100: '#e0f0e2',
  200: '#b0dbc3',
  500: '#079455',
};

export const palette: PaletteOptions = {
  mode: 'light',
  common: {
    white,
    black,
  },
  grey,
  primary: {
    main: grey[900]!,
    darker: grey[1000]!,
    dark: grey[1000]!,
    light: grey[800]!,
    lighter: grey[700]!,
    contrastText: grey[100]!,
    chip: grey[100]!,
    chipText: grey[800]!,
  },
  secondary: {
    main: grey[100]!,
    darker: grey[250]!,
    dark: grey[200]!,
    light: grey[200]!,
    lighter: grey[300]!,
    contrastText: grey[900]!,
    chip: brand[100]!,
    chipText: brand[500]!,
  },
  brand: {
    // primary in design
    main: brand[500]!,
    darker: brand[500]!,
    dark: brand[500]!,
    light: brand[500]!,
    lighter: brand[50]!,
    contrastText: grey[100]!,
    chip: brand[100]!,
    chipText: brand[500]!,
  },
  info: {
    main: white,
    dark: grey[500]!,
    darker: grey[250]!,
    light: grey[100]!,
    lighter: grey[100]!,
    contrastText: grey[800]!,
    chip: grey[200]!,
    chipText: grey[800]!,
  },
  error: {
    main: red[500]!,
    dark: red[500]!,
    darker: red[500]!,
    light: red[400]!,
    lighter: red[100]!,
    contrastText: grey[100]!,
    chip: red[100]!,
    chipText: red[500]!,
  },
  warning: {
    main: orange[500]!,
    dark: orange[500]!,
    darker: orange[500]!,
    light: orange[500]!,
    lighter: orange[100]!,
    contrastText: grey[100]!,
    chip: orange[100]!,
    chipText: orange[500]!,
  },
  success: {
    main: green[500]!,
    dark: green[500]!,
    darker: green[500]!,
    light: green[500]!,
    lighter: green[100]!,
    contrastText: grey[100]!,
    chip: red[100]!,
    chipText: brand[500]!,
  },
  text: {
    primary: grey[900]!,
    secondary: grey[700]!,
    disabled: grey[500]!,
  },
  divider: grey[250]!,
  background: {
    default: white,
    paper: white,
    area: grey[200]!,
  },
};
