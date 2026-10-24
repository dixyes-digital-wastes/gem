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

namespace Swow\Examples\Transformer\Graph;

final class DType
{
    public const F32 = 'f32';

    public const I32 = 'i32';

    public static function assertSupported(string $dtype): void
    {
        if ($dtype !== self::F32 && $dtype !== self::I32) {
            throw new GraphException("Unsupported dtype: {$dtype}");
        }
    }
}
