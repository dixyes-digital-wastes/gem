#!/usr/bin/env php
<?php

declare(strict_types=1);

require __DIR__ . '/bootstrap.php';

use Dixyes\Gem\Verification\VectorVerifier;

if ($argc !== 2) {
    fwrite(STDERR, "usage: verify.php <gemm_vectors.txt>\n");
    exit(1);
}

try {
    exit(VectorVerifier::verify($argv[1]) ? 0 : 1);
} catch (Throwable $e) {
    fwrite(STDERR, $e::class . ': ' . $e->getMessage() . "\n");
    exit(1);
}
