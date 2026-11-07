# Derleme: Visual Studio ortamını (vcvars64) yükler, CMake + Ninja ile build\ klasörüne derler.
#   .\build.ps1            # Release
#   .\build.ps1 -Debug     # hata ayıklama sürümü (build-debug\)
#   .\build.ps1 -Clean     # klasörü silip baştan
param([switch]$Debug, [switch]$Clean)
$ErrorActionPreference = 'Stop'

$vcvars = $env:EL_VCVARS
if (-not $vcvars) {
    $vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
    if (Test-Path $vswhere) {
        $vs = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
        if ($vs) { $vcvars = Join-Path $vs 'VC\Auxiliary\Build\vcvars64.bat' }
    }
}
if (-not $vcvars -or -not (Test-Path $vcvars)) {
    $vcvars = 'C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat'
}
if (-not (Test-Path $vcvars)) { throw "vcvars64.bat bulunamadi; EL_VCVARS ortam degiskeniyle yolunu verin." }

$type = if ($Debug) { 'Debug' } else { 'Release' }
$dir = if ($Debug) { 'build-debug' } else { 'build' }
Set-Location $PSScriptRoot
if ($Clean -and (Test-Path $dir)) { Remove-Item -Recurse -Force $dir }

$cuda = ''
$cudaRoot = $env:CUDA_PATH
if (-not $cudaRoot -or -not (Test-Path "$cudaRoot\bin\nvcc.exe")) {
    # CUDA_PATH henüz bu oturuma yansımamış olabilir: standart kurulum klasöründe en yeni sürümü ara
    $cudaRoot = Get-ChildItem 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v*' -Directory -ErrorAction SilentlyContinue |
        Where-Object { Test-Path "$($_.FullName)\bin\nvcc.exe" } |
        Sort-Object { [version]($_.Name.TrimStart('v')) } | Select-Object -Last 1 -ExpandProperty FullName
}
if ($cudaRoot) {
    $cuda = "-DCMAKE_CUDA_COMPILER=`"$($cudaRoot -replace '\\','/')/bin/nvcc.exe`" -DCUDAToolkit_ROOT=`"$($cudaRoot -replace '\\','/')`""
    Write-Host "CUDA: $cudaRoot"
}
$cmd = "`"$vcvars`" >nul 2>&1 && cmake -S . -B $dir -G Ninja -DCMAKE_BUILD_TYPE=$type -DCMAKE_CXX_COMPILER=cl $cuda && cmake --build $dir"
# vcvars stderr'e zararsız uyarılar yazar (ör. vswhere bulunamadı); bunlar hata sayılmasın, çıkış koduna bakılır
$ErrorActionPreference = 'Continue'
cmd /c "$cmd 2>&1"
if ($LASTEXITCODE -ne 0) { throw "derleme basarisiz ($LASTEXITCODE)" }
Write-Host "TAMAM: $dir\el.exe"
