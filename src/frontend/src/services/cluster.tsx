/* eslint-disable react-refresh/only-export-components */
import type { Dispatch, SetStateAction, FC, PropsWithChildren } from 'react';
import { createContext, useContext, useEffect, useMemo, useState } from 'react';
import { useRefCallback } from '../hooks';
import { createStreamClusterStatus, getModelList, initScheduler } from './api';
import { useHost } from './host';

import logoUrlOpenAI from '../assets/models/OpenAI-black-monoblossom.svg';
import logoUrlQwen from '../assets/models/Qwen3.png';
import logoUrlNvidia from '../assets/models/NVIDIA.png';
import logoUrlMoonshotAI from '../assets/models/MoonshotAI.png';
import logoUrlDeepseek from '../assets/models/DeepSeek.png';
import logoUrlZai from '../assets/models/Zai.png';
import logoUrlMiniMax from '../assets/models/MiniMax.png';

const logoUrlMap: Readonly<Record<string, string>> = {
  openai: logoUrlOpenAI,
  qwen: logoUrlQwen,
  nvidia: logoUrlNvidia,
  moonshotai: logoUrlMoonshotAI,
  deepseek: logoUrlDeepseek,
  zai: logoUrlZai,
  minimaxai: logoUrlMiniMax,
};

const getLogoUrl = (name: string) => {
  name = name.toLowerCase();
  const parts = name.split(/[-/]/);
  return logoUrlMap[parts[0]] || '';
};

const debugLog = (...args: any[]) => {
  console.log('%c cluster.tsx ', 'color: white; background: darkcyan;', ...args);
};

export interface ModelInfo {
  readonly name: string;
  readonly displayName: string;
  readonly logoUrl: string;

  /**
   * The VRAM required for the model in GB.
   */
  readonly vram: number;
}

export type ClusterStatus = 'idle' | 'waiting' | 'available' | 'rebalancing' | 'failed';

export interface ClusterInfo {
  readonly id: string;
  readonly status: ClusterStatus;
  readonly modelName: string;
  readonly nodeJoinCommand: Readonly<Record<string, string>>;
  readonly initNodesNumber: number;
  readonly needMoreNodes: boolean;
}

const INITIAL_CLUSTER_INFO: ClusterInfo = {
  id: '',
  status: 'idle',
  modelName: '',
  nodeJoinCommand: {},
  initNodesNumber: 4,
  needMoreNodes: false,
};

export type NodeStatus = 'waiting' | 'available' | 'failed';

export interface NodeInfo {
  readonly id: string;
  readonly status: NodeStatus;
  readonly gpuNumber: number;
  readonly gpuName: string;
  readonly gpuMemory: number;
}

// Interface

export type NetworkType = 'local' | 'remote';

export interface ClusterStates {
  readonly networkType: NetworkType;
  readonly initNodesNumber: number;
  readonly modelName: string;
  readonly modelInfo: ModelInfo | undefined;
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
  const [{ type: hostType }] = useHost();

  // Init Parameters
  const [networkType, setNetworkType] = useState<NetworkType>('local');
  const [initNodesNumber, setInitNodesNumber] = useState(1);
  const [modelName, setModelName] = useState<string>('');

  // Model List
  const [modelInfoList, setModelInfoList] = useState<readonly ModelInfo[]>([]);

  const updateModelList = useRefCallback(async () => {
    if (hostType === 'node') {
      return;
    }
    let succeed = false;
    while (!succeed) {
      try {
        const rawList = await getModelList();
        setModelInfoList((prev) => {
          const next = rawList.map<ModelInfo>(({ name, vram_gb }) => {
            name = name || '';
            vram_gb = vram_gb || 0;
            return {
              name,
              displayName: name,
              logoUrl: getLogoUrl(name),
              vram: vram_gb,
            };
          });
          if (JSON.stringify(next) !== JSON.stringify(prev)) {
            debugLog('setModelInfoList', next);
            return next;
          }
          return prev;
        });
        succeed = true;
      } catch (error) {
        console.error('getModelList error', error);
        await new Promise((resolve) => setTimeout(resolve, 2000));
      }
    }
  });

  useEffect(() => {
    updateModelList();
  }, []);

  useEffect(() => {
    if (modelInfoList.length) {
      setModelName(modelInfoList[0].name);
    }
  }, [modelInfoList]);

  // Cluster and Nodes
  const [clusterInfo, setClusterInfo] = useState<ClusterInfo>(INITIAL_CLUSTER_INFO);
  const [nodeInfoList, setNodeInfoList] = useState<readonly NodeInfo[]>([]);

  const reset = useRefCallback(() => {
    debugLog('reset');
    setClusterInfo(INITIAL_CLUSTER_INFO);
    setNodeInfoList([]);
  });

  const streamClusterStatus = useMemo(() => {
    const onMessage = (message: any) => {
      if (message.type === 'cluster_status') {
        const {
          data: {
            status,
            init_nodes_num,
            model_name,
            node_join_command,
            node_list,
            need_more_nodes,
          },
        } = message;
        setModelName((prev) => model_name || prev);
        setClusterInfo((prev) => {
          const next = {
            ...prev,
            status: (model_name && status) || 'idle',
            initNodesNumber: init_nodes_num || 0,
            modelName: model_name || '',
            nodeJoinCommand: node_join_command || {},
            needMoreNodes: need_more_nodes || false,
          };
          if (JSON.stringify(next) !== JSON.stringify(prev)) {
            debugLog('setClusterInfo', next);
            return next;
          }
          return prev;
        });
        setNodeInfoList((prev) => {
          // eslint-disable-next-line @typescript-eslint/no-explicit-any
          let next = (node_list as any[]).map<NodeInfo>(
            ({ node_id, status, gpu_num, gpu_name, gpu_memory }: any) => ({
              id: node_id,
              status,
              gpuNumber: gpu_num || 1,
              gpuName: gpu_name || '',
              gpuMemory: gpu_memory || 0,
            }),
          );

          const prevOnlineNodes = prev.filter((preNode) =>
            next.some((nextNode) => nextNode.id === preNode.id),
          );
          const prevOfflineNodes = prev
            .filter((preNode) => !next.some((nextNode) => nextNode.id === preNode.id))
            .map<NodeInfo>((offlineNode) => ({
              ...offlineNode,
              status: 'failed',
            }));

          if (JSON.stringify(next) === JSON.stringify(prevOnlineNodes)) {
            next = [...next, ...prevOfflineNodes];
          }

          if (JSON.stringify(next) !== JSON.stringify(prev)) {
            debugLog('setNodeInfoList', next);
            return next;
          }
          return prev;
        });
      }
    };
    const stream = createStreamClusterStatus({
      debugName: 'ClusterStatus',
      autoReconnect: true,
      onMessage,
      onError: reset,
    });
    return stream;
  }, []);

  useEffect(() => {
    streamClusterStatus.send();
  }, []);

  const init = useRefCallback(async () => {
    if (initNodesNumber < 1) {
      throw new Error('initNodesNumber must be greater than 0');
    }
    if (!modelName) {
      throw new Error('modelName is required');
    }

    await initScheduler({
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
        modelInfo: modelInfoList.find((model) => model.name === modelName),
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
