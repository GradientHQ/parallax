// src/router/index.tsx
import { lazy, Suspense, useEffect } from 'react';
import { useLocation, useRoutes, Navigate, useNavigate } from 'react-router-dom';
import { useCluster } from '../services';
import { useConstCallback, useRefCallback } from '../hooks';

const PATH_SETUP = '/setup';
const PATH_JOIN = '/join';
const PATH_CHAT = '/chat';

const PageSetup = lazy(() => import('../pages/setup'));
const PageJoin = lazy(() => import('../pages/join'));
const PageChat = lazy(() => import('../pages/chat'));

const debugLog = (...args: any[]) => {
  console.log('%c router.tsx ', 'color: white; background: purple;', ...args);
};

export const MainRouter = () => {
  const navigate = useNavigate();
  const { pathname } = useLocation();

  const [
    {
      clusterInfo: { status },
    },
  ] = useCluster();

  useEffect(() => {
    const lazyNavigate = (path: string) => {
      const timer = setTimeout(() => {
        debugLog('navigate to', path);
        navigate(path);
      }, 300);
      return () => clearTimeout(timer);
    };

    if (pathname === '/') {
      return lazyNavigate(PATH_SETUP);
    }
    debugLog('pathname', pathname, 'cluster status', status);
    if (status === 'idle' && pathname.startsWith(PATH_CHAT)) {
      return lazyNavigate(PATH_SETUP);
    }
    if (status === 'available' && !pathname.startsWith(PATH_CHAT)) {
      return lazyNavigate(PATH_CHAT);
    }
  }, [navigate, pathname, status]);

  const routes = useRoutes([
    {
      path: PATH_SETUP,
      element: (
        <Suspense fallback={<div>Loading...</div>}>
          <PageSetup />
        </Suspense>
      ),
    },
    {
      path: PATH_JOIN,
      element: (
        <Suspense fallback={<div>Loading...</div>}>
          <PageJoin />
        </Suspense>
      ),
    },
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
