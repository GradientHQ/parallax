import './global.css';

import type { FC, PropsWithChildren } from 'react';
import { StrictMode } from 'react';
import { HashRouter } from 'react-router-dom';
import { CssBaseline, styled } from '@mui/material';
import { ThemeProvider } from './themes';
import { MainRouter, ChatRouter } from './router';
import { ChatProvider, ClusterProvider, HostProvider, type HostProps } from './services';

const AppRoot = styled('div')(({ theme }) => {
  const { palette, typography } = theme;
  return {
    ...typography.body2,

    color: palette.text.primary,
    backgroundColor: palette.background.default,

    width: '100%',
    height: '100%',
    display: 'flex',
    flexFlow: 'column nowrap',
    justifyContent: 'center',
    alignItems: 'center',
  };
});

const Providers: FC<PropsWithChildren & { readonly hostProps: HostProps }> = ({
  children,
  hostProps,
}) => {
  return (
    <StrictMode>
      <HashRouter>
        <ThemeProvider>
          <CssBaseline />
          <AppRoot>
            <HostProvider {...hostProps}>
              <ClusterProvider>
                <ChatProvider>{children}</ChatProvider>
              </ClusterProvider>
            </HostProvider>
          </AppRoot>
        </ThemeProvider>
      </HashRouter>
    </StrictMode>
  );
};

export const Main = () => {
  return (
    <Providers hostProps={{ type: 'cluster' }}>
      <MainRouter />
    </Providers>
  );
};

export const Chat = () => {
  return (
    <Providers hostProps={{ type: 'node' }}>
      <ChatRouter />
    </Providers>
  );
};
