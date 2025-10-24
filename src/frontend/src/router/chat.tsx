// src/router/index.tsx
import { lazy, Suspense, useEffect } from 'react';
import { useLocation, useRoutes, useNavigate } from 'react-router-dom';

const PATH_CHAT = '/chat';

const PageChat = lazy(() => import('../pages/chat'));

const debugLog = (...args: any[]) => {
  console.log('%c router.tsx ', 'color: white; background: purple;', ...args);
};

export const ChatRouter = () => {
  const navigate = useNavigate();
  const { pathname } = useLocation();

  useEffect(() => {
    const lazyNavigate = (path: string) => {
      const timer = setTimeout(() => {
        debugLog('navigate to', path);
        navigate(path);
      }, 300);
      return () => clearTimeout(timer);
    };

    if (!pathname.startsWith(PATH_CHAT)) {
      return lazyNavigate(PATH_CHAT);
    }
  }, [navigate, pathname]);

  const routes = useRoutes([
    {
      path: PATH_CHAT,
      element: (
        <Suspense fallback={<div>Loading...</div>}>
          <PageChat />
        </Suspense>
      ),
    },
    {
      path: '*',
      element: <div>404 - Page Not Found</div>,
    },
  ]);
  return routes;
};
