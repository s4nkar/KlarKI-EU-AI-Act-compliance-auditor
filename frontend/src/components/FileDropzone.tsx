// Premium drag-and-drop file upload area with type + size validation.

import { useCallback } from 'react'
import { useDropzone } from 'react-dropzone'

const ACCEPTED_TYPES = {
  'application/pdf': ['.pdf'],
  'application/vnd.openxmlformats-officedocument.wordprocessingml.document': ['.docx'],
  'text/plain': ['.txt'],
  'text/markdown': ['.md'],
}
const MAX_SIZE_BYTES = 10 * 1024 * 1024

interface FileDropzoneProps {
  files: File[]
  onChange: (files: File[]) => void
}

export default function FileDropzone({ files, onChange }: FileDropzoneProps) {
  const onDrop = useCallback((accepted: File[]) => onChange(accepted), [onChange])

  const { getRootProps, getInputProps, isDragActive, fileRejections } = useDropzone({
    onDrop,
    accept: ACCEPTED_TYPES,
    maxSize: MAX_SIZE_BYTES,
    multiple: false,
  })

  const rejection = fileRejections[0]?.errors[0]

  if (files.length > 0) {
    return (
      <div className="rounded-xl border border-emerald-200 bg-emerald-50 p-4">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-lg bg-emerald-100 flex items-center justify-center shrink-0">
            <svg className="w-5 h-5 text-emerald-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
                d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
            </svg>
          </div>
          <div className="flex-1 min-w-0">
            <p className="text-sm font-semibold text-emerald-800 truncate">{files[0].name}</p>
            <p className="text-xs text-emerald-600 mt-0.5">{(files[0].size / 1024).toFixed(0)} KB · Ready to upload</p>
          </div>
          <button
            type="button"
            onClick={() => onChange([])}
            className="w-7 h-7 rounded-lg bg-emerald-100 hover:bg-emerald-200 flex items-center justify-center text-emerald-600 hover:text-emerald-800 transition-colors"
            aria-label="Remove file"
          >
            <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2.5} d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        </div>
      </div>
    )
  }

  return (
    <div>
      <div
        {...getRootProps()}
        className={`relative border-2 border-dashed rounded-xl p-10 text-center cursor-pointer transition-all duration-200 ${
          isDragActive
            ? 'border-brand-500 bg-brand-50 scale-[1.01]'
            : 'border-slate-200 hover:border-brand-400 hover:bg-slate-50'
        }`}
      >
        <input {...getInputProps()} />
        <div className="flex flex-col items-center gap-3">
          <div className={`w-14 h-14 rounded-2xl flex items-center justify-center transition-colors ${
            isDragActive ? 'bg-brand-100' : 'bg-slate-100'
          }`}>
            <svg className={`w-7 h-7 transition-colors ${isDragActive ? 'text-brand-600' : 'text-slate-400'}`}
              fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5}
                d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
            </svg>
          </div>
          {isDragActive ? (
            <p className="text-brand-600 font-semibold text-sm">Drop it here</p>
          ) : (
            <>
              <div>
                <p className="text-slate-700 font-semibold text-sm">
                  Drag &amp; drop a file, or <span className="text-brand-600 hover:text-brand-700">browse</span>
                </p>
                <p className="text-slate-400 text-xs mt-1">PDF, DOCX, TXT, MD · max 10 MB</p>
              </div>
            </>
          )}
        </div>
      </div>

      {rejection && (
        <p className="mt-2 text-sm text-red-600 flex items-center gap-1.5">
          <svg className="w-4 h-4 shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
              d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
          </svg>
          {rejection.code === 'file-too-large'
            ? 'File exceeds the 10 MB limit.'
            : rejection.code === 'file-invalid-type'
              ? 'Unsupported file type. Use .pdf, .docx, .txt, or .md.'
              : rejection.message}
        </p>
      )}
    </div>
  )
}
