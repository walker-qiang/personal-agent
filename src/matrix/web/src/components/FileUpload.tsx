import React, { useState, useRef, useCallback, DragEvent } from 'react';
import type { FileInfo } from '../types';
import { api } from '../utils/api';

interface Props {
  onFileSelected: (file: FileInfo | null) => void;
}

function formatFileSize(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

const FileUpload: React.FC<Props> = ({ onFileSelected }) => {
  const [file, setFile] = useState<FileInfo | null>(null);
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState('');
  const [dragOver, setDragOver] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  const uploadFile = useCallback(
    async (f: File) => {
      setUploading(true);
      setError('');
      try {
        const formData = new FormData();
        formData.append('file', f);

        const token = localStorage.getItem('mx_token');
        const headers: Record<string, string> = {};
        if (token) {
          headers['Authorization'] = `Bearer ${token}`;
        }

        const res = await fetch('/api/upload', {
          method: 'POST',
          headers,
          body: formData,
        });

        if (!res.ok) {
          const body = await res.json().catch(() => ({}));
          throw new Error((body as { detail?: string }).detail || `上传失败 (${res.status})`);
        }

        const data: FileInfo = await res.json();
        setFile(data);
        onFileSelected(data);
      } catch (err: unknown) {
        const msg = err instanceof Error ? err.message : '上传失败';
        setError(msg);
        onFileSelected(null);
      } finally {
        setUploading(false);
      }
    },
    [onFileSelected],
  );

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const f = e.target.files?.[0];
    if (f) {
      uploadFile(f);
    }
    // Reset input so the same file can be re-selected
    if (inputRef.current) {
      inputRef.current.value = '';
    }
  };

  const handleClick = () => {
    if (!uploading) {
      inputRef.current?.click();
    }
  };

  const handleRemove = () => {
    setFile(null);
    setError('');
    onFileSelected(null);
  };

  const handleDragOver = (e: DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragOver(true);
  };

  const handleDragLeave = (e: DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragOver(false);
  };

  const handleDrop = (e: DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragOver(false);
    const f = e.dataTransfer.files?.[0];
    if (f) {
      uploadFile(f);
    }
  };

  return (
    <div
      style={{
        ...styles.container,
        ...(dragOver ? styles.containerDragOver : {}),
      }}
      onDragOver={handleDragOver}
      onDragLeave={handleDragLeave}
      onDrop={handleDrop}
    >
      <input
        ref={inputRef}
        type="file"
        style={styles.hiddenInput}
        onChange={handleFileChange}
      />

      {file ? (
        <div style={styles.fileInfo}>
          <span style={styles.fileIcon}>{'\uD83D\uDCCE'}</span>
          <div style={styles.fileDetails}>
            <span style={styles.fileName}>{file.filename}</span>
            <span style={styles.fileSize}>{formatFileSize(file.size)}</span>
          </div>
          <button
            style={styles.removeBtn}
            onClick={handleRemove}
            title="移除文件"
            disabled={uploading}
          >
            {'\u2715'}
          </button>
        </div>
      ) : (
        <button
          style={styles.uploadBtn}
          onClick={handleClick}
          disabled={uploading}
          title="上传文件"
        >
          <span style={styles.uploadIcon}>
            {uploading ? '\u23F3' : '\uD83D\uDCCE'}
          </span>
          {uploading && <span style={styles.uploadingText}>上传中...</span>}
        </button>
      )}

      {error && (
        <div style={styles.error}>{error}</div>
      )}
    </div>
  );
};

const styles: Record<string, React.CSSProperties> = {
  container: {
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    padding: '8px 4px',
    borderRadius: 10,
    border: '2px dashed transparent',
    transition: 'border-color 0.2s, background-color 0.2s',
  },
  containerDragOver: {
    borderColor: 'var(--accent, #5b8def)',
    backgroundColor: 'rgba(91, 141, 239, 0.08)',
  },
  hiddenInput: {
    display: 'none',
  },
  uploadBtn: {
    display: 'flex',
    alignItems: 'center',
    gap: 6,
    padding: '6px 10px',
    borderRadius: 8,
    border: '1px solid var(--rule, #333)',
    backgroundColor: 'var(--bg2, #222)',
    color: 'var(--muted, #888)',
    fontSize: 13,
    cursor: 'pointer',
    transition: 'color 0.15s, border-color 0.15s',
  },
  uploadIcon: {
    fontSize: 16,
  },
  uploadingText: {
    fontSize: 12,
    color: 'var(--accent, #5b8def)',
  },
  fileInfo: {
    display: 'flex',
    alignItems: 'center',
    gap: 8,
    padding: '6px 10px',
    borderRadius: 8,
    backgroundColor: 'var(--bg2, #222)',
    border: '1px solid var(--rule, #333)',
    width: '100%',
    boxSizing: 'border-box' as const,
  },
  fileIcon: {
    fontSize: 18,
    flexShrink: 0,
  },
  fileDetails: {
    display: 'flex',
    flexDirection: 'column',
    gap: 2,
    flex: 1,
    minWidth: 0,
  },
  fileName: {
    fontSize: 12,
    fontWeight: 500,
    color: 'var(--text, #e0e0e0)',
    overflow: 'hidden',
    textOverflow: 'ellipsis',
    whiteSpace: 'nowrap',
  },
  fileSize: {
    fontSize: 11,
    color: 'var(--muted, #888)',
  },
  removeBtn: {
    width: 24,
    height: 24,
    borderRadius: 6,
    border: 'none',
    backgroundColor: 'transparent',
    color: '#ff5050',
    fontSize: 12,
    cursor: 'pointer',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    flexShrink: 0,
    transition: 'background-color 0.15s',
  },
  error: {
    marginTop: 6,
    fontSize: 11,
    color: '#ff5050',
    textAlign: 'center',
  },
};

export default FileUpload;