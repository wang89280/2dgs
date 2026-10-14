param([string]$Build,[string]$CMake='',[switch]$Install)
$ErrorActionPreference='Stop'
if(-not $Build){$Build=Join-Path ([IO.Path]::GetTempPath()) 'seed_reconstruction_final_build'}
if(-not $CMake){
    $found=Get-Command cmake -ErrorAction SilentlyContinue
    if($found){$CMake=$found.Source}else{
        $CMake='C:/Program Files/Microsoft Visual Studio/2022/Community/Common7/IDE/CommonExtensions/Microsoft/CMake/CMake/bin/cmake.exe'
    }
}
& $CMake -S $PSScriptRoot -B $Build -G 'Visual Studio 17 2022' -A x64
if($LASTEXITCODE -ne 0){throw 'CMake configure failed'}
& $CMake --build $Build --config Release --parallel 2
if($LASTEXITCODE -ne 0){throw 'CMake build failed'}
foreach($name in @('rebuild_tracks','maintain_tracks','refine_image_observations')){
    & "$Build/Release/$name.exe" --self-test
    if($LASTEXITCODE -ne 0){throw "Self-test failed: $name"}
}
if($Install){
    foreach($name in @('initialize_cameras','incremental_sfm','rebuild_tracks','maintain_tracks','refine_image_observations')){
        Copy-Item -LiteralPath "$Build/Release/$name.exe" -Destination "$PSScriptRoot/bin/$name.exe" -Force
    }
}
Write-Output 'Build and geometric self-tests complete.'
