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
  hidden?: boolean;
  branch_count?: number;
}

export interface SkillItem {
  name: string;
  description: string;
  prompt: string;
  workflow: string;
  output_format: string;
  knowledge_files?: string[];
  script_files?: string[];
}

export interface SkillFile {
  filename: string;
  content: string;
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
  message_id?: string;
  toolCalls?: ToolCall[];
  toolResults?: ToolResult[];
  agentChain?: AgentStep[];
  duration?: string;
  error?: string;
  isStreaming?: boolean;
  thinking?: string[];
  progress?: string[];
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
  is_image?: boolean;
  base64?: string;
}

// MCP types
export interface McpServer {
  name: string;
  transport: 'stdio' | 'http';
  command?: string;
  args?: string[];
  url?: string;
  env?: Record<string, string>;
  enabled: boolean;
  connected: boolean;
  tool_count?: number;
  tools?: string[];
  timeout?: number;
}

// Trace types
export interface TraceStats {
  total_events: number;
  total_sessions: number;
  total_errors: number;
}

export interface TraceSession {
  session_id: string;
  started: string;
  total_events: number;
  tool_calls: number;
  errors: number;
}

export interface TraceEvent {
  event_type: string;
  tool_name?: string;
  agent_id?: string;
  ok?: boolean;
  error?: string;
  arguments?: Record<string, unknown>;
  result_preview?: string;
  ts: string;
  elapsed_ms?: number;
}

// Session branch types
export interface BranchInfo {
  session_id: string;
  leaf_id: string;
  branched: boolean;
}

export interface BranchesResponse {
  branches: Array<{
    message_id: string;
    branch_count: number;
  }>;
}

// Confirm dialog types
export interface ConfirmAction {
  tool: string;
  args: Record<string, unknown>;
  reason: string;
}

// Model group types
export interface ModelGroup {
  label: string;
  models: string[];
}