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

namespace Swow\Examples\Transformer\Chat;

use RuntimeException;

final class ContextLimitException extends RuntimeException
{
    public function __construct(
        public readonly int $position,
        public readonly int $needed,
        public readonly int $limit,
    ) {
        parent::__construct("Context limit reached: {$position} + {$needed} > {$limit}");
    }
}
