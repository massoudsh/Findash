'use client';

import {
  ReactFlow,
  Background,
  Controls,
  BackgroundVariant,
  useNodesState,
  useEdgesState,
  type Node,
  type Edge,
  type NodeProps,
} from '@xyflow/react';
import '@xyflow/react/dist/style.css';

const COL_WIDTH = 220;
const ROW_HEIGHT = 80;
const START_X = 40;
const START_Y = 60;

/** Custom node so label text is always visible (theme foreground). Wrapper still gets node className for border/bg. */
function WorkflowNode({ data, selected }: NodeProps) {
  const label = (data?.label as string) ?? '';
  return (
    <div
      className={`px-3 py-2 min-w-[80px] text-center text-sm font-medium text-foreground whitespace-pre-wrap ${selected ? 'ring-2 ring-primary rounded-lg' : ''}`}
    >
      {label}
    </div>
  );
}

const nodeTypes = { default: WorkflowNode };

const initialNodes: Node[] = [
  // Sources
  { id: 'sources', type: 'default', position: { x: START_X, y: START_Y }, data: { label: '📡 Sources\nMarkets, News, Alt Data' }, className: '!rounded-xl !border-2 !border-amber-400/60 !bg-amber-50 dark:!bg-amber-950/30' },
  // Ingest
  { id: 'm1', type: 'default', position: { x: START_X + COL_WIDTH, y: START_Y }, data: { label: 'Nexus M1' }, className: '!rounded-lg !border !border-primary/30' },
  { id: 'm2', type: 'default', position: { x: START_X + COL_WIDTH, y: START_Y + ROW_HEIGHT }, data: { label: 'Vault M2' }, className: '!rounded-lg !border !border-primary/30' },
  { id: 'm3', type: 'default', position: { x: START_X + COL_WIDTH, y: START_Y + ROW_HEIGHT * 2 }, data: { label: 'Pulse M3' }, className: '!rounded-lg !border !border-primary/30' },
  { id: 'm9', type: 'default', position: { x: START_X + COL_WIDTH, y: START_Y + ROW_HEIGHT * 3 }, data: { label: 'Echo M9' }, className: '!rounded-lg !border !border-primary/30' },
  // Analyze
  { id: 'm5', type: 'default', position: { x: START_X + COL_WIDTH * 2, y: START_Y }, data: { label: 'Neuron M5' }, className: '!rounded-lg !border !border-chart-1/50' },
  { id: 'm7', type: 'default', position: { x: START_X + COL_WIDTH * 2, y: START_Y + ROW_HEIGHT }, data: { label: 'Oracle M7' }, className: '!rounded-lg !border !border-chart-1/50' },
  { id: 'm4', type: 'default', position: { x: START_X + COL_WIDTH * 2, y: START_Y + ROW_HEIGHT * 2 }, data: { label: 'Atlas M4' }, className: '!rounded-lg !border !border-chart-1/50' },
  { id: 'm6', type: 'default', position: { x: START_X + COL_WIDTH * 2, y: START_Y + ROW_HEIGHT * 3 }, data: { label: 'Guardian M6' }, className: '!rounded-lg !border !border-chart-1/50' },
  // Decide
  { id: 'you', type: 'default', position: { x: START_X + COL_WIDTH * 3, y: START_Y + ROW_HEIGHT * 1.5 }, data: { label: '👤 You\nApprove / Reject / Modify' }, className: '!rounded-xl !border-2 !border-amber-500 !bg-amber-100 dark:!bg-amber-900/40' },
  // Execute
  { id: 'm8', type: 'default', position: { x: START_X + COL_WIDTH * 4, y: START_Y }, data: { label: 'Shadow M8' }, className: '!rounded-lg !border !border-chart-2/50' },
  { id: 'm10', type: 'default', position: { x: START_X + COL_WIDTH * 4, y: START_Y + ROW_HEIGHT }, data: { label: 'Chronicle M10' }, className: '!rounded-lg !border !border-chart-2/50' },
  { id: 'm11', type: 'default', position: { x: START_X + COL_WIDTH * 4, y: START_Y + ROW_HEIGHT * 2 }, data: { label: 'Lens M11' }, className: '!rounded-lg !border !border-chart-2/50' },
  { id: 'reports', type: 'default', position: { x: START_X + COL_WIDTH * 4, y: START_Y + ROW_HEIGHT * 3 }, data: { label: 'Reports & Dashboards' }, className: '!rounded-lg !border !border-chart-2/50' },
];

const initialEdges: Edge[] = [
  { id: 'e-s-m1', source: 'sources', target: 'm1' },
  { id: 'e-s-m2', source: 'sources', target: 'm2' },
  { id: 'e-m1-m3', source: 'm1', target: 'm3' },
  { id: 'e-m1-m9', source: 'm1', target: 'm9' },
  { id: 'e-m3-m5', source: 'm3', target: 'm5' },
  { id: 'e-m3-m7', source: 'm3', target: 'm7' },
  { id: 'e-m9-m4', source: 'm9', target: 'm4' },
  { id: 'e-m5-m4', source: 'm5', target: 'm4' },
  { id: 'e-m7-m4', source: 'm7', target: 'm4' },
  { id: 'e-m4-m6', source: 'm4', target: 'm6' },
  { id: 'e-m6-you', source: 'm6', target: 'you' },
  { id: 'e-you-m8', source: 'you', target: 'm8' },
  { id: 'e-you-m10', source: 'you', target: 'm10' },
  { id: 'e-m8-m11', source: 'm8', target: 'm11' },
  { id: 'e-m10-m11', source: 'm10', target: 'm11' },
  { id: 'e-m11-reports', source: 'm11', target: 'reports' },
];

export function WorkflowFlowChart() {
  const [nodes, , onNodesChange] = useNodesState(initialNodes);
  const [edges, , onEdgesChange] = useEdgesState(initialEdges);

  return (
    <div className="w-full h-[480px] rounded-xl border border-border bg-muted/20 overflow-hidden">
      <ReactFlow
        nodes={nodes}
        edges={edges}
        nodeTypes={nodeTypes}
        onNodesChange={onNodesChange}
        onEdgesChange={onEdgesChange}
        fitView
        fitViewOptions={{ padding: 0.2, maxZoom: 1.2 }}
        minZoom={0.25}
        maxZoom={1.5}
        nodesDraggable
        nodesConnectable={false}
        elementsSelectable
        proOptions={{ hideAttribution: true }}
        className="text-foreground"
      >
        <Background variant={BackgroundVariant.Dots} gap={16} size={1} className="opacity-50" />
        <Controls showInteractive={false} className="!bg-card !border !border-border !rounded-lg !shadow" />
      </ReactFlow>
    </div>
  );
}
