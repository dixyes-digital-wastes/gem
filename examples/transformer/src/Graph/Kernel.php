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

interface Kernel
{
    /**
     * @param list<Tensor> $inputs
     * @param array<string, mixed> $attrs
     * @return list<float>|list<int>
     */
    public function execute(array $inputs, array $attrs): array;
}
