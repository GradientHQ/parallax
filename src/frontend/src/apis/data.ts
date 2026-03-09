export const TYPE_PATH_MAP = {
  model_list: '/api/model/list',
  scheduler_init: '/api/scheduler/init',
  node_join_command: '/api/node/join_command',
} as const;

export type TypePathMap = typeof TYPE_PATH_MAP;

export type TypeDataMap = {
  readonly model_list: [never, never];
  readonly scheduler_init: [never, never];
  readonly node_join_command: [never, never];
};
