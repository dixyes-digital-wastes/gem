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

namespace Swow\Examples\Transformer\Gguf;

final class GgufArrayRef
{
    public function __construct(
        private GgufFile $file,
        private int $offset,
        private int $elementType,
        private int $count,
    ) {
    }

    public function offset(): int
    {
        return $this->offset;
    }

    public function elementType(): int
    {
        return $this->elementType;
    }

    public function count(): int
    {
        return $this->count;
    }

    /** @return list<mixed> */
    public function toArray(): array
    {
        return $this->file->readArray($this);
    }
}
