export interface Overlays {
  shadowXSmall: string;
  shadowSmall: string;
  shadowMiddle: string;

  shadowInput: string;
  shadowInputActive: string;
  shadowInputError: string;

  buttonShadeDefault: string;
  buttonShadeActiveLight: string;
  buttonShadeActiveDark: string;
  buttonShadeBrand: string;
  buttonShadeError: string;

  cardHover: string;
}

declare module '@mui/material' {
  interface Theme {
    overlays: Overlays;
  }

  interface ThemeOptions {
    overlays: Overlays;
  }
}

export const overlays: Overlays = {
  shadowXSmall: '0px 1px 2px 0px rgba(16, 24, 40, 0.05)',
  shadowSmall: '0px 1px 2px 0px rgba(49, 49, 48, 0.05)',
  shadowMiddle:
    '0px 8px 8px -4px rgba(52, 51, 51, 0.03), 0px 20px 24px -4px rgba(52, 51, 51, 0.08)',

  // shadowInput: '0px 1px 2px 0px rgba(49, 49, 48, 0.05)',
  shadowInput: 'none',
  shadowInputActive:
    '0px 0px 0px 4px rgba(49, 49, 48, 0.05), 0px 1px 2px 0px rgba(49, 49, 48, 0.05)',
  shadowInputError:
    '0px 0px 0px 4px rgba(221, 82, 76, 0.12), 0px 1px 2px 0px rgba(221, 82, 76, 0.05)',

  buttonShadeDefault: '0px 1px 2px 0px rgba(49, 49, 48, 0.05)',
  buttonShadeActiveLight:
    '0px 0px 0px 4px rgba(49, 49, 48, 0.05), 0px 1px 2px 0px rgba(49, 49, 48, 0.05)',
  buttonShadeActiveDark:
    '0px 0px 0px 4px rgba(49, 49, 48, 0.24), 0px 1px 2px 0px rgba(49, 49, 48, 0.32)',
  buttonShadeBrand: '0 1px 2px 0 rgba(7, 139, 89, 0.10), 0 0 0 4px rgba(7, 139, 89, 0.10)',
  buttonShadeError:
    '0px 0px 0px 4px rgba(221, 82, 76, 0.12), 0px 1px 2px 0px rgba(221, 82, 76, 0.05)',

  cardHover: '0px 4px 16px 0px rgba(160, 160, 160, 0.25)',
};
