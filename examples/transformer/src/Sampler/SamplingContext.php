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

namespace Swow\Examples\Transformer\Sampler;

final class SamplingContext
{
    /** @param list<int> $history */
    public function __construct(
        public array $history = [],
        public int $position = 0,
        public ?RandomSource $random = null,
    ) {
    }
}
