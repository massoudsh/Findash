import { FinetuningJob } from '@/components/models/llm-finetuning'; // Assuming the type is exported

const mockJobs: FinetuningJob[] = [
  { id: 'job_123', modelName: 'BART-large-fine-tuned-v1', status: 'completed', createdAt: '2023-10-26T10:00:00Z', completedAt: '2023-10-26T14:30:00Z' },
  { id: 'job_456', modelName: 'PEFT-output-new', status: 'running', createdAt: '2023-10-27T11:00:00Z' },
  { id: 'job_789', modelName: 'old-model-archive', status: 'failed', createdAt: '2023-10-25T09:00:00Z', completedAt: '2023-10-25T10:00:00Z' },
];

export async function getFinetuningJobs(): Promise<FinetuningJob[]> {
  console.log("Mock API: Fetching fine-tuning jobs data.");
  return new Promise(resolve => {
    setTimeout(() => resolve(mockJobs), 500);
  });
}

export async function triggerFinetuning(outputDir: string): Promise<{ taskId: string }> {
  console.log(`Mock API: Triggering fine-tuning for output directory: ${outputDir}`);
  return new Promise(resolve => {
    setTimeout(() => {
      resolve({ taskId: `task_${Date.now()}` });
    }, 800);
  });
}

export async function getJobStatus(taskId: string) {
  console.log(`Mock API: Fetching job status for task: ${taskId}`);
  return new Promise(resolve => {
    setTimeout(() => {
      resolve({
        id: taskId,
        status: 'running',
        progress: Math.floor(Math.random() * 100),
        message: 'Training in progress...'
      });
    }, 300);
  });
} 