<?php

declare(strict_types=1);

namespace Dixyes\Gem\Tests;

use Dixyes\Gem\Graph\Weight;
use Dixyes\Gem\Thread\CooperativeBarrier;
use Dixyes\Gem\Thread\CooperativeContext;
use Dixyes\Gem\Thread\Linear;
use Swow\SharedBuffer;
use Swow\Sync\ThreadBarrier;

final class GemvWorker
{
    /** @param list<float> $input */
    public static function run(
        Weight $weight,
        array $input,
        int $rank,
        int $threads,
        SharedBuffer $outputA,
        SharedBuffer $outputB,
        CooperativeBarrier $barrier,
    ): void {
        $context = new CooperativeContext(
            $rank,
            $threads,
            $outputA,
            $outputB,
            new SharedBuffer(8),
            new SharedBuffer(8),
            new SharedBuffer(8),
            $barrier,
            new ThreadBarrier($threads),
        );
        CooperativeContext::set($context);
        try {
            Linear::matvec($weight, $input);
        } finally {
            CooperativeContext::set(null);
        }
    }
}
