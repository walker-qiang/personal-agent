export interface Provider {
  id: string;
  label: string;
  models: string[];
}

export interface ImageModel {
  id: string;
  provider: string;
  label: string;
}

export interface VideoModel {
  id: string;
  provider: string;
  label: string;
}

export interface SessionItem {
  id: string;
  title: string;
  created_at: string;
  updated_at: string;
  turns: number;
}

export interface SkillItem {
  name: string;
  description: string;
  prompt: string;
  workflow: string;
  output_format: string;
}

export interface AgentStep {
  agent: string;
  task: string;
  status: 'pending' | 'running' | 'done' | 'error';
  result?: string;
}

export interface ToolCall {
  id: string;
  name: string;
  arguments: Record<string, unknown>;
}

export interface ToolResult {
  id: string;
  name: string;
  result: unknown;
  error?: string;
  duration_ms?: number;
}

export interface Message {
  id: string;
  role: 'user' | 'assistant' | 'system';
  content: string;
  toolCalls?: ToolCall[];
  toolResults?: ToolResult[];
  agentChain?: AgentStep[];
  duration?: string;
  error?: string;
  isStreaming?: boolean;
}

export interface SSEEvent {
  type: string;
  data: Record<string, unknown>;
}

export interface FileInfo {
  file_id: string;
  filename: string;
  mime_type: string;
  size: number;
}