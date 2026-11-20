# PowerShell chunked curriculum runner
# Usage: .\run_chunk_curriculum.ps1
# Env vars you can set before running:
#   FILES_PER_CHUNK=5
#   MAX_TEXTS=2000000
#   TRAIN_STEPS_PER_CHUNK=50000
#   USE_TEXT_CACHE=1

$ErrorActionPreference = 'Stop'

$BASE = $PSScriptRoot
$PYTHON = "C:\Users\thoma\.lmstudio\apps\bionic\projects\d49037d8-47f4-5808-9028-c707de117f8f\workspace\TRANSLATION_OS_V1\.venv\Scripts\python.exe"
$TRAINER = "scripts\train_weave_decoder_gpu.py"

$FILES_PER_CHUNK = if ($env:FILES_PER_CHUNK) { [int]$env:FILES_PER_CHUNK } else { 5 }
$MAX_TEXTS = if ($env:MAX_TEXTS) { $env:MAX_TEXTS } else { "2000000" }
$TRAIN_STEPS_PER_CHUNK = if ($env:TRAIN_STEPS_PER_CHUNK) { [int]$env:TRAIN_STEPS_PER_CHUNK } else { 50000 }
$USE_TEXT_CACHE = if ($env:USE_TEXT_CACHE) { $env:USE_TEXT_CACHE } else { "1" }

# Count parquet files
$parquetFiles = Get-ChildItem -Path "$BASE\..\trainingData" -Filter *.parquet -Recurse | Sort-Object Name
$totalFiles = $parquetFiles.Count
$totalChunks = [Math]::Ceiling($totalFiles / $FILES_PER_CHUNK)

Write-Host "Total parquet files: $totalFiles, files per chunk: $FILES_PER_CHUNK, total chunks: $totalChunks"

for ($i = 0; $i -lt $totalChunks; $i++) {
    Write-Host "=== Chunk $i / $($totalChunks-1) ==="
    $env:CHUNK_INDEX = $i
    $env:FILES_PER_CHUNK = $FILES_PER_CHUNK
    $env:MAX_TEXTS = $MAX_TEXTS
    $env:USE_TEXT_CACHE = $USE_TEXT_CACHE
    $env:TRAIN_STEPS = $TRAIN_STEPS_PER_CHUNK

    Write-Host "Starting chunk $i with TRAIN_STEPS=$TRAIN_STEPS_PER_CHUNK"
    & $PYTHON $TRAINER
    if ($LASTEXITCODE -ne 0) {
        Write-Warning "Trainer exited with code $LASTEXITCODE. Continuing to next chunk."
    }
    # Optional mastery check could be added here, e.g., parse last val loss
    Write-Host "Chunk $i complete. Proceeding to next chunk..."
}

Write-Host "All chunks processed."
