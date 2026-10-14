param(
    [string]$Images='',
    [string]$Output='',
    [string]$Python='',
    [ValidateSet('cuda','cpu')][string]$Provider,
    [string]$Calibration='',
    [int]$Iterations,
    [int]$Width,
    [int]$Threads,
    [int]$MaxKeypoints,
    [ValidateSet('sequential','exhaustive')][string]$Pairing,
    [int]$Window,
    [int]$InitialPairs,
    [double]$StationaryRadiusPx,
    [string]$Distro,
    [string]$WslPython,
    [switch]$Eval,
    [switch]$SparseOnly,
    [switch]$SkipRender,
    [switch]$Mesh,
    [switch]$KeepIntermediates,
    [switch]$TrackRecovery,
    [ValidateSet('off','localization','weighted','learned','multiview')][string]$ImageRefinement,
    [ValidateSet('off','robust')][string]$PointRefinement,
    [ValidateSet('off','view-consensus')][string]$TrackPartition,
    [switch]$CheckEnv,
    [switch]$DryRun
)
$ErrorActionPreference='Stop'
if(-not $Python){
    $candidate=Join-Path $env:USERPROFILE 'Miniconda3/envs/colmap/python.exe'
    if(Test-Path -LiteralPath $candidate){$Python=$candidate}else{$Python='python'}
}
# Only explicit overrides are forwarded; defaults live in config.json.
$arguments=@('-B',"$PSScriptRoot/run_pipeline.py")
$valueFlags=@{
    Images='--images'; Output='--output'; Calibration='--calibration'
    Provider='--provider'; Iterations='--iterations'; Width='--width'
    Threads='--threads'; MaxKeypoints='--max-keypoints'; Pairing='--pairing'
    Window='--window'; InitialPairs='--initial-pairs'; StationaryRadiusPx='--stationary-radius-px'
    Distro='--distro'; WslPython='--wsl-python'; ImageRefinement='--image-refinement'
    PointRefinement='--point-refinement'
    TrackPartition='--track-partition'
}
foreach($name in $valueFlags.Keys){
    if($PSBoundParameters.ContainsKey($name)){
        $value=[System.Convert]::ToString($PSBoundParameters[$name],[System.Globalization.CultureInfo]::InvariantCulture)
        $arguments+=@($valueFlags[$name],$value)
    }
}
$switchFlags=@{Eval='eval'; SparseOnly='sparse-only'; SkipRender='skip-render'; Mesh='mesh'; KeepIntermediates='keep-intermediates'; TrackRecovery='track-recovery'}
foreach($name in $switchFlags.Keys){
    if($PSBoundParameters.ContainsKey($name)){
        if($PSBoundParameters[$name]){$arguments+='--'+$switchFlags[$name]}
        else{$arguments+='--no-'+$switchFlags[$name]}
    }
}
if($CheckEnv){$arguments+='--check-env'}
if($DryRun){$arguments+='--dry-run'}
& $Python @arguments
if($LASTEXITCODE -ne 0){throw "Pipeline failed (exit $LASTEXITCODE). See result logs if execution started."}
