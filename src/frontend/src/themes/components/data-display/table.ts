import type { Components, Theme } from '@mui/material';
import { Paper, tableCellClasses, tableRowClasses } from '@mui/material';

export const MuiTableContainer = (theme: Theme): Components<Theme>['MuiTableContainer'] => {
  return {
    defaultProps: {
      component: Paper,
    },
    styleOverrides: {},
  };
};

export const MuiTable = (theme: Theme): Components<Theme>['MuiTable'] => {
  return {
    defaultProps: {},
    styleOverrides: {},
  };
};

export const MuiTableHead = (theme: Theme): Components<Theme>['MuiTableHead'] => {
  const { palette } = theme;
  return {
    defaultProps: {},
    styleOverrides: {
      root: {},
    },
  };
};

export const MuiTableBody = (theme: Theme): Components<Theme>['MuiTableBody'] => {
  return {
    defaultProps: {},
    styleOverrides: {
      root: {},
    },
  };
};

export const MuiTableRow = (theme: Theme): Components<Theme>['MuiTableRow'] => {
  const { palette } = theme;

  return {
    defaultProps: {},
    styleOverrides: {
      root: {
        [`&:hover, &.${tableRowClasses.hover}:hover`]: {
          backgroundColor: palette.grey[100],
        },
        [`&.${tableRowClasses.selected}`]: {
          backgroundColor: palette.grey[50],
          '&:hover': {
            backgroundColor: palette.grey[200],
          },
        },
        [`&:last-child>.${tableCellClasses.body}`]: {
          border: 0,
        },
      },
    },
  };
};

export const MuiTableCell = (theme: Theme): Components<Theme>['MuiTableCell'] => {
  const { palette, typography } = theme;

  return {
    defaultProps: {},
    styleOverrides: {
      root: {
        padding: '1rem',
        borderBottom: `1px solid ${palette.divider}`,
        variants: [
          {
            props: { variant: 'head' },
            style: {
              backgroundColor: palette.background.area,
            },
          },
          {
            props: ({ ownerState }) => !!ownerState.stickyHeader,
            style: {
              backgroundColor: palette.background.area,
            },
          },
        ],
      },
    },
  };
};

export const MuiTableFooter = (theme: Theme): Components<Theme>['MuiTableFooter'] => {
  return {
    defaultProps: {},
    styleOverrides: {},
  };
};

export const MuiTablePagination = (theme: Theme): Components<Theme>['MuiTablePagination'] => {
  return {
    defaultProps: {},
    styleOverrides: {},
  };
};

export const MuiTablePaginationActions = (
  theme: Theme,
): Components<Theme>['MuiTablePaginationActions'] => {
  return {
    defaultProps: {},
    styleOverrides: {},
  };
};

export const MuiTableSortLabel = (theme: Theme): Components<Theme>['MuiTableSortLabel'] => {
  return {
    defaultProps: {},
    styleOverrides: {},
  };
};
