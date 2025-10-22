import './global.css';

import type { FC, PropsWithChildren } from 'react';
import { StrictMode } from 'react';
import { HashRouter } from 'react-router-dom';
import { CssBaseline, styled } from '@mui/material';
import { ThemeProvider } from './themes';
import { MainRouter, ChatRouter } from './router';
import { ChatProvider, ClusterProvider } from './services';

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

const Providers: FC<PropsWithChildren> = ({ children }) => {
  return (
    <StrictMode>
      <HashRouter>
        <ThemeProvider>
          <CssBaseline />
          <AppRoot>
            <ClusterProvider>
              <ChatProvider>{children}</ChatProvider>
            </ClusterProvider>
          </AppRoot>
        </ThemeProvider>
      </HashRouter>
    </StrictMode>
  );
};

export const Main = () => {
  return (
    <Providers>
      <MainRouter />
    </Providers>
  );
};

export const Chat = () => {
  return (
    <Providers>
      <ChatRouter />
    </Providers>
  );
};
