// src/router/index.tsx
import { lazy, Suspense } from 'react';
import { useRoutes, Navigate, useNavigate } from 'react-router-dom';

const PageSetup = lazy(() => import('../pages/setup'));
const PageJoin = lazy(() => import('../pages/join'));
const PageChat = lazy(() => import('../pages/chat'));

export const Router = () => {
  const navigate = useNavigate();

  const routes = useRoutes([
    {
      path: '/',
      element: <Navigate to='/setup' replace />, // redirect to the page setup
    },
    {
      path: '/setup',
      element: (
        <Suspense fallback={<div>Loading...</div>}>
          <PageSetup />
        </Suspense>
      ),
    },
    {
      path: '/join',
      element: (
        <Suspense fallback={<div>Loading...</div>}>
          <PageJoin />
        </Suspense>
      ),
    },
    {
      path: '/chat',
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
