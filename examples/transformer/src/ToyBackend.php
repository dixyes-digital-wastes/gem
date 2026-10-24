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
use function pack;
use function unpack;

/*
 * Bridge to the optional toy extension.
 *
 * Callers check enabled() before any toy symbol is touched, so the demo keeps
 * working without the extension. F32 weight tensors are cached per Weight
 * object and row range; the weak map releases them with the model
 */
final class ToyBackend
{
    public const MAX_RANK = 4;

    private const DTYPE = 'f32';

    /** @var WeakMap<Weight, array<string, Tensor>>|null */
    private static ?WeakMap $weights = null;

    private static ?bool $enabled = null;

    private function __construct()
    {
    }

    public static function enabled(): bool
    {
        return self::$enabled ??= extension_loaded('toy') && class_exists('Toy\\Tensor');
    }

    /**
     * @param list<float>|list<int> $values
     * @param list<int> $shape
     */
    public static function vector(array $values, array $shape): Tensor
    {
        return Tensor::fromBytes(self::DTYPE, $shape, pack('g*', ...$values));
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
        self::$weights ??= new WeakMap();
        $key = $begin . ':' . $end;
        $tensors = self::$weights[$weight] ?? [];
        if (isset($tensors[$key])) {
            return $tensors[$key];
        }
        $bytes = $begin === 0 && $end === $weight->rows() && !$weight->isLazy()
            ? $weight->data()
            : $weight->slice($begin, $end);
        $tensors[$key] = Tensor::fromBytes(self::DTYPE, [$end - $begin, $weight->columns()], $bytes);
        self::$weights[$weight] = $tensors;

        return $tensors[$key];
    }
}
