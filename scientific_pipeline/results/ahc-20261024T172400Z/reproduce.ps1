param(
    [Parameter(Mandatory=$true)][string]$RepoCheckout,
    [Parameter(Mandatory=$true)][string]$Python,
    [Parameter(Mandatory=$true)][string]$Dataset,
    [Parameter(Mandatory=$true)][string]$BaselineManifest,
    [Parameter(Mandatory=$true)][string]$NewRunRoot
)
$ErrorActionPreference='Stop'
$expectedPipeline='38503dbe8422982deeb3680bbc3e82fd950da4e1'
if ((git -C $RepoCheckout rev-parse HEAD) -ne $expectedPipeline) {throw 'Require the exact published pipeline checkout'}
if (git -C $RepoCheckout status --porcelain) {throw 'Require clean code checkout'}
if (Test-Path -LiteralPath $NewRunRoot) {throw 'Output root already exists; use a new destination'}
$protocol=Join-Path $RepoCheckout 'scientific_pipeline/experiments/ahc-v1.json'
$oldPythonPath=$env:PYTHONPATH
try {
    $env:PYTHONPATH=Join-Path $RepoCheckout 'scientific_pipeline/src'
    $parentManifest=$null
    foreach ($stage in @('train','validate','test','closeout')) {
        $stageOutput=Join-Path $NewRunRoot $stage
        $arguments=@('-m','siar_science',"ahc-$stage",'--dataset',$Dataset,'--baseline-manifest',$BaselineManifest,'--protocol',$protocol,'--output',$stageOutput,'--pipeline-sha',$expectedPipeline)
        if ($parentManifest) {$arguments+=@('--parent-manifest',$parentManifest)}
        & $Python @arguments
        if ($LASTEXITCODE -ne 0) {throw "Stage $stage failed or stopped; preserve its manifest and inspect evidence"}
        $parentManifest=Join-Path $stageOutput 'manifest.json'
        if ((Get-Content -LiteralPath $parentManifest -Raw | ConvertFrom-Json).status -ne 'SUCCESS') {throw "Unsuccessful stage $stage"}
    }
}
finally {$env:PYTHONPATH=$oldPythonPath}
