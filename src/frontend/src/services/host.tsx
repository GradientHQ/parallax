/* eslint-disable react-refresh/only-export-components */
import type { FC, PropsWithChildren } from 'react';
import { createContext, useContext, useMemo } from 'react';
import { useConst } from '../hooks';

export type HostType = 'cluster' | 'node';

export interface HostProps {
  readonly type: HostType;
}

// eslint-disable-next-line @typescript-eslint/no-empty-object-type
export interface HostStates extends HostProps {}

// eslint-disable-next-line @typescript-eslint/no-empty-object-type
export interface HostActions {}

const context = createContext<readonly [HostStates, HostActions] | undefined>(undefined);

const { Provider } = context;

export const HostProvider: FC<PropsWithChildren<HostProps>> = ({ children, type }) => {
  const actions: HostActions = useConst(() => ({}));

  const value = useMemo<readonly [HostStates, HostActions]>(
    () => [
      {
        type,
      },
      actions,
    ],
    [type, actions],
  );

  return <Provider value={value}>{children}</Provider>;
};

export const useHost = (): readonly [HostStates, HostActions] => {
  const value = useContext(context);
  if (!value) {
    throw new Error('useHost must be used within a HostProvider');
  }
  return value;
};
