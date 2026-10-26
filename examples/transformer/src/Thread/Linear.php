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

namespace Swow\Examples\Transformer\Thread;

use RuntimeException;
use Swow\Examples\Gemm\Kernel\F32Gemv;
use Swow\Examples\Gemm\Quantization\Q4_0;
use Swow\Examples\Gemm\Quantization\Q8_0;
use Swow\Examples\Transformer\Graph\Weight;
use Swow\Examples\Transformer\ToyBackend;
use Toy\Tensor;

use function array_push;
use function intdiv;
use function min;
use function Toy\Tensor\matvec_f32;
use function Toy\Tensor\matvec_q8_0_q8_0;
use function Toy\Tensor\quantize_q8_0;

/*
 * Linear kernels.
 *
 * Inside a cooperative worker, a linear op splits its rows, writes its own
 * range into the shared output buffer, waits at the op barrier, then reads the
 * full result. Without a context (single-thread use, tests), it computes the
 * whole result locally.
 *
 * The activation and the result stay in the extension when the caller asks for
 * tensor results, so a chain of graph ops does not convert them back to PHP
 * arrays between the ops
 */
final class Linear
{
    private const MIN_PARALLEL_WORK = 1_000_000;

    private const BARRIER_TIMEOUT = 60_000;

    /**
     * @param list<float>|Tensor $input
     * @return list<float>|Tensor
     */
    public static function matvec(Weight $weight, array|object $input, bool $asTensor = false): array|object
    {
        return self::matvecMany([$weight], $input, $asTensor)[0];
    }

    /**
     * @param list<Weight> $weights
     * @param list<float>|Tensor $input
     * @return list<list<float>|Tensor>
     */
    public static function matvecMany(array $weights, array|object $input, bool $asTensor = false): array
    {
        $context = CooperativeContext::current();
        if ($context === null || !self::canSplit($weights) || self::work($weights) < self::MIN_PARALLEL_WORK) {
            $result = [];
            $operands = [];
            foreach ($weights as $weight) {
                $columns = $weight->columns();
                $result[] = self::computeRange(
                    $weight,
                    0,
                    $weight->rows(),
                    $input,
                    $asTensor,
                    ToyBackend::enabled()
                        ? ($operands[$columns . ':' . $weight->type()] ??= self::operand(
                            $input,
                            $columns,
                            $weight->type() === Weight::Q8_0,
                        ))
                        : null,
                );
            }

            return $result;
        }

        /* A tensor result is written as f32 bytes, a PHP result as doubles */
        $stride = $asTensor ? 4 : 8;
        $buffer = $context->outputBuffer($context->opIndex());
        $offset = 0;
        $ranges = [];
        $operands = [];
        foreach ($weights as $weight) {
            $rows = $weight->rows();
            $columns = $weight->columns();
            $begin = intdiv($rows * $context->rank, $context->threads);
            $end = intdiv($rows * ($context->rank + 1), $context->threads);
            $values = self::computeRange(
                $weight,
                $begin,
                $end,
                $input,
                $asTensor,
                ToyBackend::enabled()
                    ? ($operands[$columns . ':' . $weight->type()] ??= self::operand(
                        $input,
                        $columns,
                        $weight->type() === Weight::Q8_0,
                    ))
                    : null,
            );
            if ($values instanceof Tensor) {
                $buffer->write($offset + $begin * 4, $values->toBytes());
            } else {
                $buffer->writeF64($offset + $begin * 8, $values);
            }
            $ranges[] = [$offset, $rows];
            $offset += $rows * $stride;
        }

        if (!$context->opBarrier->wait(self::BARRIER_TIMEOUT)) {
            throw new RuntimeException('Linear op barrier timed out');
        }

        $result = [];
        foreach ($ranges as [$base, $rows]) {
            $result[] = $asTensor
                ? ToyBackend::fromBytes($buffer->read($base, $rows * 4), [$rows])
                : $buffer->viewF64($base, $rows);
        }
        $context->advanceOp();

        return $result;
    }

    /** @param list<Weight> $weights */
    private static function canSplit(array $weights): bool
    {
        foreach ($weights as $weight) {
            if ($weight->tensorName() === null || $weight->sourcePath() === null) {
                return false;
            }
        }

        return true;
    }

    /** @param list<Weight> $weights */
    private static function work(array $weights): int
    {
        $work = 0;
        foreach ($weights as $weight) {
            $work += $weight->rows() * $weight->columns();
        }

        return $work;
    }

    /**
     * @param list<float>|Tensor $input
     * @return list<float>|Tensor
     */
    private static function computeRange(
        Weight $weight,
        int $begin,
        int $end,
        array|object $input,
        bool $asTensor,
        ?Tensor $operand = null,
    ): array|object
    {
        $rows = $end - $begin;
        if ($rows === 0) {
            return $asTensor ? ToyBackend::fromBytes('', [$rows]) : [];
        }
        $columns = $weight->columns();
        $q8 = $weight->type() === Weight::Q8_0;
        if (!$q8 && $weight->type() !== Weight::F32) {
            return self::scalar($weight->slice($begin, $end), $weight->type(), $input, $rows, $columns, $asTensor);
        }

        /* A generated kernel rejects more than MAX_KERNEL_ROWS rows, so a wide
         * range runs as ordered chunks instead of one scalar fallback. The
         * activation is quantized once and every chunk reads the same vector */
        $values = [];
        $bytes = '';
        for ($start = $begin; $start < $end; $start += ToyBackend::MAX_KERNEL_ROWS) {
            $stop = min($end, $start + ToyBackend::MAX_KERNEL_ROWS);
            $chunkRows = $stop - $start;
            if ($q8) {
                $tensor = ToyBackend::q8LinearWeight($weight, $start, $stop);
                if ($tensor === null) {
                    return self::scalar($weight->slice($begin, $end), Weight::Q8_0, $input, $rows, $columns, $asTensor);
                }
                $operand ??= self::operand($input, $columns, true);
                $chunk = matvec_q8_0_q8_0($tensor, $operand, $chunkRows, $columns);
            } else {
                $tensor = ToyBackend::linearWeight($weight, $start, $stop);
                if ($tensor === null) {
                    return self::scalar($weight->slice($begin, $end), Weight::F32, $input, $rows, $columns, $asTensor);
                }
                $operand ??= self::operand($input, $columns, false);
                $chunk = matvec_f32($tensor, $operand, $chunkRows, $columns);
            }
            if ($asTensor) {
                $bytes .= $chunk->toBytes();
            } else {
                array_push($values, ...ToyBackend::floats($chunk));
            }
        }

        return $asTensor ? ToyBackend::fromBytes($bytes, [$rows]) : $values;
    }

    /**
     * Quantized or f32 activation tensor of a linear input
     *
     * A resident tensor is used as it is; a plain array is converted once and
     * every row chunk reads the same operand
     *
     * @param list<float>|Tensor $input
     */
    private static function operand(array|object $input, int $columns, bool $q8): Tensor
    {
        if ($q8) {
            if ($input instanceof Tensor && ToyBackend::has('quantize_q8_0')) {
                return quantize_q8_0($input);
            }
            $values = $input instanceof Tensor ? ToyBackend::floats($input) : $input;

            return ToyBackend::has('quantize_q8_0')
                ? ToyBackend::q8Activation($values, $columns)
                : ToyBackend::q8Vector(Q8_0::quantize($values), $columns);
        }

        return $input instanceof Tensor ? $input : ToyBackend::vector($input, [$columns]);
    }

    /**
     * @param list<float>|Tensor $input
     * @return list<float>|Tensor
     */
    private static function scalar(string $data, string $type, array|object $input, int $rows, int $columns, bool $asTensor): array|object
    {
        $values = $input instanceof Tensor ? ToyBackend::floats($input) : $input;
        $result = self::compute($data, $type, $values, $rows, $columns);

        return $asTensor ? ToyBackend::vector($result, [$rows]) : $result;
    }

    /**
     * @param list<float> $input
     * @return list<float>
     */
    private static function compute(string $data, string $type, array $input, int $rows, int $columns): array
    {
        return match ($type) {
            Weight::Q8_0 => Q8_0::matvec($data, $input, $rows, $columns),
            Weight::Q4_0 => Q4_0::matvec($data, $input, $rows, $columns),
            Weight::F32 => F32Gemv::matvec($data, $input, $rows, $columns),
            default => throw new RuntimeException('Unsupported weight type: ' . $type),
        };
    }
}
