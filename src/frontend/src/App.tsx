import { styled } from '@mui/material';
import { Router } from './router';
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

export const App = () => {
  return (
    <AppRoot>
      <ClusterProvider>
        <ChatProvider>
          <Router />
        </ChatProvider>
      </ClusterProvider>
    </AppRoot>
  );
};
