/* eslint-disable react-refresh/only-export-components */
import type { Dispatch, SetStateAction, FC, PropsWithChildren } from 'react';
import { createContext, useContext, useEffect, useMemo, useState } from 'react';
import { useRefCallback } from '../hooks';

export interface ModelInfo {
  readonly name: string;
  readonly displayName: string;
  readonly logoUrl: string;
}

export type ClusterStatus = 'idle' | 'waiting' | 'available' | 'rebalancing';

export interface ClusterInfo {
  readonly id: string;
  readonly status: ClusterStatus;
  readonly nodeJoinCommand: string;
  readonly initNodesNumber: number;
}

export type NodeStatus = 'waiting' | 'available' | 'failed';

export interface NodeInfo {
  readonly id: string;
  readonly status: NodeStatus;
  readonly gpuName: string;
  readonly gpuMemory: number;
}

// Interface

export type NetworkType = 'local' | 'remote';

export interface ClusterStates {
  readonly networkType: NetworkType;
  readonly initNodesNumber: number;
  readonly modelName: string;
  readonly modelInfoList: readonly ModelInfo[];

  readonly clusterInfo: ClusterInfo;
  readonly nodeInfoList: readonly NodeInfo[];
}

export interface ClusterActions {
  readonly setNetworkType: Dispatch<SetStateAction<NetworkType>>;
  readonly setInitNodesNumber: Dispatch<SetStateAction<number>>;
  readonly setModelName: Dispatch<SetStateAction<string>>;

  readonly init: () => Promise<void>;
}

// Implementation

const context = createContext<readonly [ClusterStates, ClusterActions] | undefined>(undefined);

const { Provider } = context;

export const ClusterProvider: FC<PropsWithChildren> = ({ children }) => {
  // Init Parameters
  const [networkType, setNetworkType] = useState<NetworkType>('local');
  const [initNodesNumber, setInitNodesNumber] = useState(2);
  const [modelName, setModelName] = useState('gpt-4o');

  // Model List
  const [modelInfoList, setModelInfoList] = useState<readonly ModelInfo[]>([]);
  useEffect(() => {
    // TODO: fetch api get model list
    setModelInfoList([
      // MOCK
      { name: 'gpt-4o', displayName: 'GPT-4o', logoUrl: '' },
      { name: 'gpt-4o-mini', displayName: 'GPT-4o Mini', logoUrl: '' },
      { name: 'gpt-4o-turbo', displayName: 'GPT-4o Turbo', logoUrl: '' },
    ]);
  }, []);
  useEffect(() => {
    if (modelInfoList.length) {
      setModelName(modelInfoList[0].name);
    }
  }, [modelInfoList]);

  // Cluster and Nodes
  const [clusterInfo, setClusterInfo] = useState<ClusterInfo>(() => ({
    id: '',
    status: 'idle',
    nodeJoinCommand: 'parallax join 192.168.1.100',
    initNodesNumber: 4,
  }));
  const [nodeInfoList, setNodeInfoList] = useState<readonly NodeInfo[]>(() => [
    // MOCK
    {
      id: 'sfasge235rytdfgq35q346234wedfss',
      status: 'available',
      gpuName: 'NVIDIA A100',
      gpuMemory: 24,
    },
    {
      id: 'dfgshjldkrewi25246esfdgsh345sdf',
      status: 'waiting',
      gpuName: 'NVIDIA A100',
      gpuMemory: 24,
    },
    {
      id: 'dfgberiuiwuyhy25346tea2342sdf12',
      status: 'failed',
      gpuName: 'NVIDIA A100',
      gpuMemory: 24,
    },
  ]);

  const init = useRefCallback(async () => {
    // TODO: fetch api init scheduler
    setClusterInfo((prev) => ({
      ...prev,
      status: 'waiting',
    }));
  });

  const actions: ClusterActions = useMemo(() => {
    return {
      setNetworkType,
      setInitNodesNumber,
      setModelName,
      init,
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const value = useMemo<readonly [ClusterStates, ClusterActions]>(
    () => [
      {
        networkType,
        initNodesNumber,
        modelName,
        modelInfoList,
        clusterInfo,
        nodeInfoList,
      },
      actions,
    ],
    [networkType, initNodesNumber, modelName, modelInfoList, clusterInfo, nodeInfoList, actions],
  );

  return <Provider value={value}>{children}</Provider>;
};

export const useCluster = (): readonly [ClusterStates, ClusterActions] => {
  const value = useContext(context);
  if (!value) {
    throw new Error('useCluster must be used within a ClusterProvider');
  }
  return value;
};
