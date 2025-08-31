/**
 * Composant DropZone pour l'upload de fichiers
 * Drag & drop avec preview et validation
 */

'use client'

import * as React from 'react'
import { cn } from '@/lib/utils'
import { Button } from '@/components/ui/button'
import { formatFileSize } from '@/lib/utils'

// ============================================
// Types
// ============================================

export interface DropZoneProps {
  onFilesSelected: (files: File[]) => void
  acceptedFormats?: string[]
  maxFileSize?: number // in bytes
  maxFiles?: number
  multiple?: boolean
  disabled?: boolean
  className?: string
  showPreview?: boolean
}

interface FileError {
  file: File
  error: string
}

// ============================================
// Constants
// ============================================

const DEFAULT_ACCEPTED_FORMATS = ['.pdf', '.doc', '.docx', '.xls', '.xlsx', '.txt']
const DEFAULT_MAX_FILE_SIZE = 50 * 1024 * 1024 // 50MB
const DEFAULT_MAX_FILES = 10

// ============================================
// Icons
// ============================================

const Icons = {
  upload: (
    <svg className="w-12 h-12" fill="none" viewBox="0 0 24 24" stroke="currentColor">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
    </svg>
  ),
  file: (
    <svg className="w-8 h-8" fill="none" viewBox="0 0 24 24" stroke="currentColor">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
    </svg>
  ),
  check: (
    <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
    </svg>
  ),
  x: (
    <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
    </svg>
  ),
}

// ============================================
// DropZone Component
// ============================================

export const DropZone: React.FC<DropZoneProps> = ({
  onFilesSelected,
  acceptedFormats = DEFAULT_ACCEPTED_FORMATS,
  maxFileSize = DEFAULT_MAX_FILE_SIZE,
  maxFiles = DEFAULT_MAX_FILES,
  multiple = true,
  disabled = false,
  className,
  showPreview = true,
}) => {
  const [isDragging, setIsDragging] = React.useState(false)
  const [files, setFiles] = React.useState<File[]>([])
  const [errors, setErrors] = React.useState<FileError[]>([])
  const fileInputRef = React.useRef<HTMLInputElement>(null)
  const dragCounter = React.useRef(0)

  // Validate single file
  const validateFile = (file: File): string | null => {
    // Check file type
    const fileExtension = `.${file.name.split('.').pop()?.toLowerCase()}`
    if (!acceptedFormats.some(format => fileExtension === format.toLowerCase())) {
      return `Invalid file type. Accepted formats: ${acceptedFormats.join(', ')}`
    }

    // Check file size
    if (file.size > maxFileSize) {
      return `File size exceeds ${formatFileSize(maxFileSize)}`
    }

    return null
  }

  // Process selected files
  const processFiles = (fileList: FileList | File[]) => {
    const filesArray = Array.from(fileList)
    const validFiles: File[] = []
    const fileErrors: FileError[] = []

    // Check max files limit
    if (filesArray.length > maxFiles) {
      fileErrors.push({
        file: filesArray[0],
        error: `Maximum ${maxFiles} files allowed`,
      })
      setErrors(fileErrors)
      return
    }

    // Validate each file
    filesArray.forEach(file => {
      const error = validateFile(file)
      if (error) {
        fileErrors.push({ file, error })
      } else {
        validFiles.push(file)
      }
    })

    // Update state
    setFiles(validFiles)
    setErrors(fileErrors)

    // Notify parent if valid files exist
    if (validFiles.length > 0) {
      onFilesSelected(validFiles)
    }
  }

  // Handle file input change
  const handleFileInput = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      processFiles(e.target.files)
    }
  }

  // Handle drag events
  const handleDragEnter = (e: React.DragEvent) => {
    e.preventDefault()
    e.stopPropagation()
    dragCounter.current++
    if (e.dataTransfer.items && e.dataTransfer.items.length > 0) {
      setIsDragging(true)
    }
  }

  const handleDragLeave = (e: React.DragEvent) => {
    e.preventDefault()
    e.stopPropagation()
    dragCounter.current--
    if (dragCounter.current === 0) {
      setIsDragging(false)
    }
  }

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault()
    e.stopPropagation()
  }

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault()
    e.stopPropagation()
    setIsDragging(false)
    dragCounter.current = 0

    if (disabled) return

    const droppedFiles = e.dataTransfer.files
    if (droppedFiles && droppedFiles.length > 0) {
      processFiles(droppedFiles)
    }
  }

  // Remove file from selection
  const removeFile = (index: number) => {
    const newFiles = files.filter((_, i) => i !== index)
    setFiles(newFiles)
    if (newFiles.length > 0) {
      onFilesSelected(newFiles)
    }
  }

  // Clear all files
  const clearFiles = () => {
    setFiles([])
    setErrors([])
    if (fileInputRef.current) {
      fileInputRef.current.value = ''
    }
  }

  // Get file type icon/color
  const getFileTypeStyle = (fileName: string) => {
    const ext = fileName.split('.').pop()?.toLowerCase()
    switch (ext) {
      case 'pdf':
        return { color: 'text-red-500', bgColor: 'bg-red-50' }
      case 'doc':
      case 'docx':
        return { color: 'text-blue-500', bgColor: 'bg-blue-50' }
      case 'xls':
      case 'xlsx':
        return { color: 'text-green-500', bgColor: 'bg-green-50' }
      default:
        return { color: 'text-gray-500', bgColor: 'bg-gray-50' }
    }
  }

  return (
    <div className={cn('w-full', className)}>
      {/* Drop Zone Area */}
      <div
        onDragEnter={handleDragEnter}
        onDragLeave={handleDragLeave}
        onDragOver={handleDragOver}
        onDrop={handleDrop}
        onClick={() => !disabled && fileInputRef.current?.click()}
        className={cn(
          'relative rounded-xl border-2 border-dashed transition-all duration-200 cursor-pointer',
          'hover:border-primary hover:bg-primary/5',
          isDragging && 'border-primary bg-primary/10 scale-[1.02]',
          disabled && 'opacity-50 cursor-not-allowed hover:border-border hover:bg-transparent',
          !isDragging && !disabled && 'border-muted-foreground/25',
          'p-8 sm:p-12'
        )}
      >
        {/* Hidden file input */}
        <input
          ref={fileInputRef}
          type="file"
          multiple={multiple}
          accept={acceptedFormats.join(',')}
          onChange={handleFileInput}
          className="hidden"
          disabled={disabled}
        />

        {/* Drop zone content */}
        <div className="flex flex-col items-center justify-center text-center">
          <div className={cn(
            'mb-4 rounded-full p-4 transition-colors',
            isDragging ? 'bg-primary/20 text-primary' : 'bg-muted text-muted-foreground'
          )}>
            {Icons.upload}
          </div>

          <h3 className="text-lg font-semibold mb-2">
            {isDragging ? 'Drop files here' : 'Drop files or click to upload'}
          </h3>

          <p className="text-sm text-muted-foreground mb-4">
            {multiple ? `Up to ${maxFiles} files, ` : 'Single file, '}
            max {formatFileSize(maxFileSize)} each
          </p>

          <div className="flex flex-wrap items-center justify-center gap-2 text-xs text-muted-foreground">
            <span>Accepted formats:</span>
            {acceptedFormats.map((format, index) => (
              <span
                key={index}
                className="px-2 py-1 bg-muted rounded-md font-mono"
              >
                {format}
              </span>
            ))}
          </div>

          {!disabled && (
            <Button
              variant="outline"
              size="sm"
              className="mt-4"
              onClick={(e) => {
                e.stopPropagation()
                fileInputRef.current?.click()
              }}
            >
              Browse Files
            </Button>
          )}
        </div>

        {/* Dragging overlay */}
        {isDragging && (
          <div className="absolute inset-0 rounded-xl bg-primary/10 backdrop-blur-sm flex items-center justify-center">
            <div className="text-primary text-center">
              <div className="mb-2">{Icons.upload}</div>
              <p className="font-semibold">Release to upload</p>
            </div>
          </div>
        )}
      </div>

      {/* File preview */}
      {showPreview && files.length > 0 && (
        <div className="mt-6 space-y-3">
          <div className="flex items-center justify-between">
            <h4 className="text-sm font-medium">
              Selected files ({files.length})
            </h4>
            <Button
              variant="ghost"
              size="sm"
              onClick={clearFiles}
              className="text-muted-foreground hover:text-foreground"
            >
              Clear all
            </Button>
          </div>

          <div className="space-y-2">
            {files.map((file, index) => {
              const fileStyle = getFileTypeStyle(file.name)
              return (
                <div
                  key={index}
                  className="flex items-center gap-3 p-3 rounded-lg border bg-card hover:bg-accent/5 transition-colors group"
                >
                  <div className={cn(
                    'flex-shrink-0 w-10 h-10 rounded-lg flex items-center justify-center',
                    fileStyle.bgColor
                  )}>
                    <span className={fileStyle.color}>{Icons.file}</span>
                  </div>

                  <div className="flex-1 min-w-0">
                    <p className="text-sm font-medium truncate">
                      {file.name}
                    </p>
                    <p className="text-xs text-muted-foreground">
                      {formatFileSize(file.size)}
                    </p>
                  </div>

                  <Button
                    variant="ghost"
                    size="icon-sm"
                    onClick={(e) => {
                      e.stopPropagation()
                      removeFile(index)
                    }}
                    className="opacity-0 group-hover:opacity-100 transition-opacity"
                  >
                    {Icons.x}
                  </Button>
                </div>
              )
            })}
          </div>
        </div>
      )}

      {/* Errors */}
      {errors.length > 0 && (
        <div className="mt-4 p-4 rounded-lg bg-destructive/10 border border-destructive/20">
          <p className="text-sm font-medium text-destructive mb-2">
            Upload errors:
          </p>
          <ul className="space-y-1">
            {errors.map((error, index) => (
              <li key={index} className="text-sm text-destructive/80">
                • {error.file.name}: {error.error}
              </li>
            ))}
          </ul>
        </div>
      )}

      {/* Success message */}
      {files.length > 0 && errors.length === 0 && (
        <div className="mt-4 p-4 rounded-lg bg-success/10 border border-success/20">
          <div className="flex items-center gap-2 text-success">
            {Icons.check}
            <p className="text-sm font-medium">
              {files.length} {files.length === 1 ? 'file' : 'files'} ready for upload
            </p>
          </div>
        </div>
      )}
    </div>
  )
}

export default DropZone