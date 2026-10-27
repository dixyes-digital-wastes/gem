<?php

declare(strict_types=1);

namespace Dixyes\Gem\Tests;

use Dixyes\Gem\Support\CpuAffinity;
use PHPUnit\Framework\TestCase;

use const PHP_OS_FAMILY;

/**
 * @internal
 */
final class CpuAffinityTest extends TestCase
{
    public function testDarwinAffinityClaimIsUsable(): void
    {
        if (PHP_OS_FAMILY !== 'Darwin') {
            self::markTestSkipped('macOS only');
        }
        if (!CpuAffinity::isAvailable()) {
            $this->addToAssertionCount(1);

            return;
        }

        CpuAffinity::pin(0);
        $this->addToAssertionCount(1);
    }
}
