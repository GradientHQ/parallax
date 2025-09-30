import { createHttpStreamFactory } from './http-stream';

export const API_BASE_URL = import.meta.env.DEV ? 'http://0.0.0.0:3001' : '';

export const getModelList = async (): Promise<readonly string[]> => {
  while (true) {
    try {
      const response = await fetch(`${API_BASE_URL}/model/list`, { method: 'GET' });
      const message = await response.json();
      if (message.type !== 'model_list') {
        throw new Error(`Invalid message type: ${message.type}.`);
      }
      return message.data;
    } catch (error) {
      console.error('getModelList error', error);
      await new Promise((resolve) => setTimeout(resolve, 2000));
    }
  }
};

export const initScheduler = async (params: {
  model_name: string;
  init_nodes_num: number;
  is_local_network: boolean;
}): Promise<void> => {
  const response = await fetch(`${API_BASE_URL}/scheduler/init`, {
    method: 'POST',
    body: JSON.stringify(params),
  });
  const message = await response.json();
  if (message.type !== 'scheduler_init') {
    throw new Error(`Invalid message type: ${message.type}.`);
  }
  return message.data;
};

export const createStreamClusterStatus = createHttpStreamFactory({
  url: `${API_BASE_URL}/cluster/status`,
  method: 'GET',
});
