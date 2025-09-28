/* eslint-disable react-refresh/only-export-components */
import type { Dispatch, SetStateAction, FC, PropsWithChildren } from 'react';
import { createContext, useContext, useEffect, useMemo, useState } from 'react';
import { useRefCallback } from '../hooks';
import { createStreamClusterStatus, getModelList, initScheduler } from './api';

import logoUrlQwen from '../assets/models/qwen.png';

const debugLog = (...args: any[]) => {
  console.log('%c cluster.tsx ', 'color: white; background: darkcyan;', ...args);
};

export interface ModelInfo {
  readonly name: string;
  readonly displayName: string;
  readonly logoUrl: string;
}

export type ClusterStatus = 'idle' | 'waiting' | 'available' | 'rebalancing';

export interface ClusterInfo {
  readonly id: string;
  readonly status: ClusterStatus;
  readonly nodeJoinCommand: Readonly<Record<string, string>>;
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
  const [initNodesNumber, setInitNodesNumber] = useState(1);
  const [modelName, setModelName] = useState<string>('');

  // Model List
  const [modelInfoList, setModelInfoList] = useState<readonly ModelInfo[]>([]);
  useEffect(() => {
    getModelList().then((modelList) => {
      setModelInfoList(
        modelList.map((name) => ({
          name,
          displayName: name,
          logoUrl: logoUrlQwen,
        })),
      );
    });
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
    nodeJoinCommand: {},
    initNodesNumber: 4,
  }));
  const [nodeInfoList, setNodeInfoList] = useState<readonly NodeInfo[]>(() => [
    // MOCK
    // {
    //   id: 'sfasge235rytdfgq35q346234wedfss',
    //   status: 'available',
    //   gpuName: 'NVIDIA A100',
    //   gpuMemory: 24,
    // },
    // {
    //   id: 'dfgshjldkrewi25246esfdgsh345sdf',
    //   status: 'waiting',
    //   gpuName: 'NVIDIA A100',
    //   gpuMemory: 24,
    // },
    // {
    //   id: 'dfgberiuiwuyhy25346tea2342sdf12',
    //   status: 'failed',
    //   gpuName: 'NVIDIA A100',
    //   gpuMemory: 24,
    // },
  ]);

  const streamClusterStatus = useMemo(() => {
    const onMessage = (message: any) => {
      if (message.type === 'cluster_status') {
        const {
          data: { status, init_nodes_num, model_name, node_join_command, node_list },
        } = message;
        setClusterInfo((prev) => {
          const next = {
            ...prev,
            status,
            initNodesNumber: init_nodes_num || 0,
            modelName: model_name || '',
            nodeJoinCommand: node_join_command || {},
          };
          if (JSON.stringify(next) !== JSON.stringify(prev)) {
            debugLog('setClusterInfo', next);
            return next;
          }
          return prev;
        });
        setNodeInfoList((prev) => {
          // eslint-disable-next-line @typescript-eslint/no-explicit-any
          const next = node_list.map(({ node_id, status, gpu_name, gpu_memory }: any) => ({
            id: node_id,
            status,
            gpuName: gpu_name,
            gpuMemory: gpu_memory,
          }));
          if (JSON.stringify(next) !== JSON.stringify(prev)) {
            debugLog('setNodeInfoList', next);
            return next;
          }
          return prev;
        });
      }
    };
    const stream = createStreamClusterStatus({
      onMessage,
    });
    stream.send();
    return stream;
  }, []);

  const init = useRefCallback(async () => {
    initScheduler({
      model_name: modelName,
      init_nodes_num: initNodesNumber,
      is_local_network: networkType === 'local',
    });
    // setClusterInfo((prev) => ({
    //   ...prev,
    //   status: 'waiting',
    // }));
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
