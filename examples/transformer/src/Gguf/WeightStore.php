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

interface WeightStore
{
    public function tensor(string $name): TensorInfo;

    public function read(string $name, int $offset, int $length): string;

    public function row(string $name, int $row): string;

    public function rows(string $name, int $rowBegin, int $rowEnd): string;
}
