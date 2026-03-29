// Drag-and-drop file upload area with type + size validation.

import { useCallback } from 'react'
import { useDropzone } from 'react-dropzone'

const ACCEPTED_TYPES = {
  'application/pdf': ['.pdf'],
  'application/vnd.openxmlformats-officedocument.wordprocessingml.document': ['.docx'],
  'text/plain': ['.txt'],
  'text/markdown': ['.md'],
}
const MAX_SIZE_BYTES = 10 * 1024 * 1024 // 10 MB

interface FileDropzoneProps {
  files: File[]
  onChange: (files: File[]) => void
}

export default function FileDropzone({ files, onChange }: FileDropzoneProps) {
  const onDrop = useCallback((accepted: File[]) => {
    onChange(accepted)
  }, [onChange])

  const { getRootProps, getInputProps, isDragActive, fileRejections } = useDropzone({
    onDrop,
    accept: ACCEPTED_TYPES,
    maxSize: MAX_SIZE_BYTES,
    multiple: false,
  })

  const rejection = fileRejections[0]?.errors[0]

  return (
    <div>
      <div
        {...getRootProps()}
        className={`border-2 border-dashed rounded-xl p-8 text-center cursor-pointer transition-colors
          ${isDragActive
            ? 'border-brand-500 bg-brand-50'
            : 'border-slate-300 hover:border-brand-400 hover:bg-slate-50'
          }`}
      >
        <input {...getInputProps()} />
        <div className="flex flex-col items-center gap-2">
          <svg className="w-10 h-10 text-slate-300" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5}
              d="M9 13h6m-3-3v6m5 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
          </svg>
          {isDragActive ? (
            <p className="text-brand-600 font-medium">Drop the file here</p>
          ) : (
            <>
              <p className="text-slate-600 font-medium">Drag &amp; drop a file, or click to select</p>
              <p className="text-slate-400 text-sm">.pdf · .docx · .txt · .md · max 10 MB</p>
            </>
          )}
        </div>
      </div>

      {rejection && (
        <p className="mt-2 text-sm text-red-600">
          {rejection.code === 'file-too-large'
            ? 'File exceeds the 10 MB limit.'
            : rejection.code === 'file-invalid-type'
              ? 'Unsupported file type. Use .pdf, .docx, .txt, or .md.'
              : rejection.message}
        </p>
      )}

      {files.length > 0 && (
        <div className="mt-3 flex items-center gap-2 text-sm text-slate-600">
          <svg className="w-4 h-4 text-green-500 shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
          </svg>
          <span className="truncate font-medium">{files[0].name}</span>
          <span className="text-slate-400 shrink-0">
            ({(files[0].size / 1024).toFixed(0)} KB)
          </span>
          <button
            type="button"
            onClick={e => { e.stopPropagation(); onChange([]) }}
            className="ml-auto text-slate-400 hover:text-red-500 transition-colors"
            aria-label="Remove file"
          >
            ×
          </button>
        </div>
      )}
    </div>
  )
}
