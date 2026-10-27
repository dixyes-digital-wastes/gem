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

use function abs;
use function class_exists;
use function getenv;
use function ini_get;
use function intdiv;
use function substr;
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
            if (getenv('GEM_EXPECT_SWOW') === 'true') {
                self::fail('Swow is not loaded although the job requires it');
            }
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

        self::assertSameFloats($expected, $outputA->viewF64(0, $rows), self::f32Delta($weight, $input));
    }

    public function testToyGeneratedKernels(): void
    {
        if (!ToyBackend::enabled() || (int) ini_get('toy.kernel_arena_size') === 0) {
            if (getenv('GEM_EXPECT_TOY') === 'true') {
                self::fail('Toy is not loaded or its kernel arena is disabled');
            }
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
    private static function assertSameFloats(array $expected, array $actual, ?callable $delta = null): void
    {
        self::assertCount(count($expected), $actual);
        foreach ($expected as $index => $value) {
            self::assertEqualsWithDelta(
                $value,
                $actual[$index],
                $delta === null ? 1e-5 : $delta($index, $value),
                "row {$index}",
            );
        }
    }

    /*
    * A f32 row rounds its whole dot product, so the error follows the summed
    * magnitude of the terms rather than the magnitude of the result
    */
    private static function f32Delta(Weight $weight, array $input): ?callable
    {
        if (!ToyBackend::enabled()) {
            return null;
        }
        $data = $weight->data();
        $blocks = intdiv($weight->columns(), Q8_0::BLOCK_SIZE);
        $blockBytes = $blocks * Q8_0::BLOCK_BYTES;

        return static function (int $row, float $expected) use ($data, $input, $blockBytes): float {
            $magnitude = 0.0;
            foreach (Q8_0::dequantize(substr($data, $row * $blockBytes, $blockBytes)) as $index => $value) {
                $magnitude += abs($value * $input[$index]);
            }

            return max(1e-6, abs($expected) * 1e-6, $magnitude * 1e-7);
        };
    }
}
