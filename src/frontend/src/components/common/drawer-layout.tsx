import type { FC, PropsWithChildren } from 'react';
import { Stack, styled } from '@mui/material';

const DrawerLayoutRoot = styled(Stack)(({ theme }) => {
  const { spacing } = theme;
  return {
    width: '100%',
    height: '100%',
    overflow: 'hidden',
  };
});

export const DrawerLayout: FC<PropsWithChildren> = ({ children }) => {
  return <DrawerLayoutRoot>{children}</DrawerLayoutRoot>;
};
