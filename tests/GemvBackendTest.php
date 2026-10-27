<?php

declare(strict_types=1);

namespace Dixyes\Gem\Tests;

use PHPUnit\Framework\TestCase;
use Dixyes\Gem\Graph\Weight;
use Dixyes\Gem\Quantization\Q8_0;
use Dixyes\Gem\Support\Threads;
use Dixyes\Gem\Thread\CooperativeBarrier;
use Dixyes\Gem\Thread\CooperativeContext;
use Dixyes\Gem\Thread\Linear;
use Dixyes\Gem\ToyBackend;
use Swow\SharedBuffer;
use Swow\Sync\ThreadBarrier;

use function class_exists;
use function ini_get;
use function intdiv;
use function Toy\Tensor\kernel_stats;

final class GemvBackendTest extends TestCase
{
    public function testNativeGemv(): void
    {
        $weight = self::weight(64, 64);
        $input = self::input(64);
        $expected = Q8_0::matvec($weight->data(), $input, 64, 64);

        self::assertSameFloats($expected, Linear::matvec($weight, $input));
    }

    public function testSwowMultithreadedGemv(): void
    {
        if (!class_exists(\Swow\Thread::class)) {
            $this->markTestSkipped('Swow is not loaded');
        }

        /* Above Linear::MIN_PARALLEL_WORK so the call splits its rows */
        $rows = 8_192;
        $columns = 128;
        $threads = 2;
        $weight = new Weight(
            'cooperative',
            Weight::Q8_0,
            $rows,
            $columns,
            Q8_0::generate(intdiv($rows * $columns, 32), 20261102),
            'synthetic.gguf',
            'cooperative',
        );
        $input = self::input($columns);
        $expected = Q8_0::matvec($weight->data(), $input, $rows, $columns);

        $outputA = new SharedBuffer($rows * 8);
        $outputB = new SharedBuffer($rows * 8);
        $barrier = new CooperativeBarrier($threads);
        $workers = [];
        for ($rank = 0; $rank < $threads; $rank++) {
            $workers[] = Threads::spawn(GemvWorker::class . '::run', [
                $weight,
                $input,
                $rank,
                $threads,
                $outputA,
                $outputB,
                $barrier,
            ]);
        }
        Threads::join($workers);

        self::assertSameFloats($expected, $outputA->viewF64(0, $rows));
    }

    public function testToyGeneratedKernels(): void
    {
        if (!ToyBackend::enabled() || (int) ini_get('toy.kernel_arena_size') === 0) {
            $this->markTestSkipped('Toy generated kernels are not enabled');
        }
        $weight = self::weight(4, 64);
        $input = self::input(64);
        kernel_stats(true);
        Linear::matvec($weight, $input);
        $stats = kernel_stats();
        if (!$stats['enabled'] || !$stats['avx2_available'] || $stats['generated'] === 0) {
            $this->markTestSkipped('No generated kernel is available on this CPU');
        }
        self::assertSame(0, $stats['generation_failures']);
        self::assertSame(0, $stats['scalar_calls']);
    }

    private static function weight(int $rows, int $columns): Weight
    {
        return Weight::packed(
            'native',
            Weight::Q8_0,
            $rows,
            $columns,
            Q8_0::generate(intdiv($rows * $columns, 32), 20261102),
        );
    }

    /** @return list<float> */
    private static function input(int $columns): array
    {
        $values = [];
        for ($index = 0; $index < $columns; $index++) {
            $values[] = (($index * 29) % 23 - 11) / 8.0;
        }

        return $values;
    }

    /** @param list<float> $expected @param list<float> $actual */
    private static function assertSameFloats(array $expected, array $actual): void
    {
        self::assertCount(count($expected), $actual);
        foreach ($expected as $index => $value) {
            self::assertEqualsWithDelta($value, $actual[$index], 1e-5, "row {$index}");
        }
    }
}
