<?php

/**
 * This file is part of Swow
 *
 * @link    https://github.com/swow/swow
 * @contact twosee <twosee@php.net>
 *
 * For the full copyright and license information,
 * please view the LICENSE file that was distributed with this source code
 */

declare(strict_types=1);

require dirname(__DIR__) . '/autoload.php';

use Swow\Examples\Gemm\Verification\VectorVerifier;

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
