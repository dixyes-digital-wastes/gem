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

namespace Swow\Examples\Transformer;

use Swow\Examples\Transformer\Graph\Weight;
use Toy\Tensor;
use WeakMap;

use function array_values;
use function class_exists;
use function extension_loaded;
use function function_exists;
use function pack;
use function Toy\Tensor\quantize_q8_0;
use function unpack;

/*
 * Bridge to the optional toy extension.
 *
 * Callers check enabled() before any toy symbol is touched, so the demo keeps
 * working without the extension. F32 and q8_0 weight tensors are cached per
 * Weight object and row range; the weak maps release them with the model
 */
final class ToyBackend
{
    public const MAX_RANK = 4;

    /* A generated kernel rejects a row count above 65535 and the call falls
     * back to the scalar path, so a wider range runs as row chunks */
    public const MAX_KERNEL_ROWS = 65_535;

    private const F32_DTYPE = 'f32';

    private const Q8_DTYPE = 'q8_0';

    /** @var WeakMap<Weight, array<string, Tensor>>|null */
    private static ?WeakMap $f32Weights = null;

    /** @var WeakMap<Weight, array<string, Tensor>>|null */
    private static ?WeakMap $q8Weights = null;

    private static ?bool $enabled = null;

    /** @var array<string, bool> */
    private static array $operators = [];

    private function __construct()
    {
    }

    public static function enabled(): bool
    {
        return self::$enabled ??= extension_loaded('toy') && class_exists('Toy\\Tensor');
    }

    /** True when the loaded extension provides the named operator */
    public static function has(string $operator): bool
    {
        return self::$operators[$operator] ??= self::enabled()
            && function_exists('Toy\\Tensor\\' . $operator);
    }

    /**
     * @param list<float>|list<int> $values
     * @param list<int> $shape
     */
    public static function vector(array $values, array $shape): Tensor
    {
        return Tensor::fromBytes(self::F32_DTYPE, $shape, pack('g*', ...$values));
    }

    /**
     * Quantize an f32 activation row in the extension
     *
     * The extension follows ggml: it divides by the f16 rounded scale and
     * clamps to the int8 range, while Q8_0::quantize() divides by the f32
     * scale in double and keeps -128, so a code can differ by one
     *
     * @param list<float> $values
     */
    public static function q8Activation(array $values, int $columns): Tensor
    {
        return quantize_q8_0(self::vector($values, [$columns]));
    }

    /** Return a q8_0 vector tensor from the bytes of Q8_0::quantize() */
    public static function q8Vector(string $bytes, int $columns): Tensor
    {
        return Tensor::fromBytes(self::Q8_DTYPE, [$columns], $bytes);
    }

    /** @return list<float> */
    public static function floats(Tensor $tensor): array
    {
        return array_values(unpack('g*', $tensor->toBytes()));
    }

    /** Return the cached f32 tensor for a row range of an f32 weight, null otherwise */
    public static function linearWeight(Weight $weight, int $begin, int $end): ?Tensor
    {
        if (!self::enabled() || $weight->type() !== Weight::F32) {
            return null;
        }
        return self::cached(self::$f32Weights ??= new WeakMap(), $weight, $begin, $end, self::F32_DTYPE);
    }

    /** Return the cached q8_0 tensor for a row range of a q8_0 weight, null otherwise */
    public static function q8LinearWeight(Weight $weight, int $begin, int $end): ?Tensor
    {
        if (!self::enabled() || $weight->type() !== Weight::Q8_0) {
            return null;
        }
        return self::cached(self::$q8Weights ??= new WeakMap(), $weight, $begin, $end, self::Q8_DTYPE);
    }

    /** @param WeakMap<Weight, array<string, Tensor>> $cache */
    private static function cached(WeakMap $cache, Weight $weight, int $begin, int $end, string $dtype): Tensor
    {
        $key = $begin . ':' . $end;
        $tensors = $cache[$weight] ?? [];
        if (isset($tensors[$key])) {
            return $tensors[$key];
        }
        $bytes = $begin === 0 && $end === $weight->rows() && !$weight->isLazy()
            ? $weight->data()
            : $weight->slice($begin, $end);
        $tensors[$key] = Tensor::fromBytes($dtype, [$end - $begin, $weight->columns()], $bytes);
        $cache[$weight] = $tensors;

        return $tensors[$key];
    }
}
