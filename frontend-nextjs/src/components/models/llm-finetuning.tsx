export interface FinetuningJob {
  id: string;
  modelName: string;
  status: 'running' | 'completed' | 'failed' | 'pending';
  createdAt: string;
  completedAt?: string;
}
