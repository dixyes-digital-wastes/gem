#!/usr/bin/env php
<?php

declare(strict_types=1);

require __DIR__ . '/bootstrap.php';

use Dixyes\Gem\Bench\GemmBench;
use Dixyes\Gem\Support\CpuAffinity;

function usage(): void
{
    fwrite(STDERR, <<<'TXT'
usage: gemm.php [size] [threads] [--efficiency]

  size         matrix dimension, default 512
  threads      worker threads, default physical cores
  --efficiency also run a single-thread pass and report the speedup

TXT);
    exit(1);
}

$arguments = [];
$efficiency = false;
foreach (array_slice($argv, 1) as $argument) {
    if ($argument === '--efficiency') {
        $efficiency = true;
        continue;
    }
    if (str_starts_with($argument, '--')) {
        usage();
    }
    $arguments[] = $argument;
}
if (count($arguments) > 2) {
    usage();
}

$size = (int) ($arguments[0] ?? 512);
$threads = max(1, (int) ($arguments[1] ?? count(CpuAffinity::physicalCpus('auto'))));

try {
    GemmBench::run($size, $threads, $efficiency);
} catch (Throwable $e) {
    fwrite(STDERR, $e::class . ': ' . $e->getMessage() . "\n");
    exit(1);
}
