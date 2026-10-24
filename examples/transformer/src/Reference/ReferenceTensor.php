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

namespace Swow\Examples\Transformer\Reference;

final class ReferenceTensor
{
    /** @param list<int> $dimensions @param list<float> $data */
    public function __construct(
        private string $name,
        private array $dimensions,
        private array $data,
    ) {
    }

    public function name(): string
    {
        return $this->name;
    }

    /** @return list<int> */
    public function dimensions(): array
    {
        return $this->dimensions;
    }

    /** @return list<float> */
    public function data(): array
    {
        return $this->data;
    }
}
