<?php

declare(strict_types=1);

namespace Dixyes\Gem\Tests;

use PHPUnit\Framework\TestCase;
use Dixyes\Gem\Quantization\Q8_0;
use Dixyes\Gem\Graph\Weight;
use Dixyes\Gem\Thread\Linear;
use Dixyes\Gem\ToyBackend;

use function ini_get;
use function intdiv;
use function pack;
use function substr;
use function Toy\Tensor\kernel_enable;
use function Toy\Tensor\kernel_stats;

/**
 * @internal
 */
final class LinearChunkTest extends TestCase
{
    /* One row count above the 65535 row generated kernel limit */
    private const ROWS = 70_000;

    private const COLUMNS = 64;

    public function testQ8MatvecAboveKernelRowCap(): void
    {
        $weight = $this->weight(self::ROWS);
        $input = $this->input();
        $kernels = $this->kernelsGenerate();
        if ($kernels) {
            kernel_stats(true);
        }

        $logits = Linear::matvec($weight, $input);

        $stats = $kernels ? kernel_stats() : null;
        $this->assertCount(self::ROWS, $logits);

        /* Every sampled row must match a single row call, so a wrong chunk
         * boundary or order shows up here */
        $rowBytes = $weight->rowBytes();
        foreach ([0, 1, 65_533, 65_534, 65_535, 65_536, self::ROWS - 2, self::ROWS - 1] as $row) {
            $single = Weight::packed(
                'row',
                Weight::Q8_0,
                1,
                self::COLUMNS,
                substr($weight->data(), $row * $rowBytes, $rowBytes),
            );
            $this->assertSame(
                pack('g', Linear::matvec($single, $input)[0]),
                pack('g', $logits[$row]),
                "row {$row}",
            );
        }

        if ($stats === null) {
            return;
        }

        /* Above the cap the whole range must still run as generated kernels */
        $this->assertSame(0, $stats['generation_failures']);
        $this->assertSame(0, $stats['scalar_calls']);
        $this->assertGreaterThanOrEqual(2, $stats['kernel_calls']);

        /* The generated kernels and the scalar fallback are bit-identical for
         * q8_0 x q8_0, so the chunked result has to match the scalar path */
        kernel_enable(false);
        try {
            $scalar = Linear::matvec($weight, $input);
        } finally {
            kernel_enable(true);
        }
        $this->assertSame($scalar, $logits);
    }

    private function weight(int $rows): Weight
    {
        return Weight::packed(
            'chunk',
            Weight::Q8_0,
            $rows,
            self::COLUMNS,
            Q8_0::generate($rows * intdiv(self::COLUMNS, Q8_0::BLOCK_SIZE), 20261101),
        );
    }

    /** @return list<float> */
    private function input(): array
    {
        $input = [];
        for ($index = 0; $index < self::COLUMNS; $index++) {
            $input[] = (($index * 29) % 23 - 11) / 8.0;
        }

        return $input;
    }

    private function kernelsGenerate(): bool
    {
        if (!ToyBackend::enabled() || (int) ini_get('toy.kernel_arena_size') === 0) {
            return false;
        }
        $stats = kernel_stats();
        /* The q8_0 x q8_0 matvec has no scalar generated kernel */
        if (!$stats['enabled'] || !$stats['avx2_available']) {
            return false;
        }
        kernel_stats(true);
        Linear::matvec($this->weight(4), $this->input());
        $stats = kernel_stats();

        return $stats['generated'] > 0 && $stats['generation_failures'] === 0 && $stats['scalar_calls'] === 0;
    }
}
