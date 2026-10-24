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

final class KernelRegistry
{
    /** @var array<string, Kernel> */
    private array $kernels = [];

    public function register(string $name, Kernel $kernel): void
    {
        $this->kernels[$name] = $kernel;
    }

    public function get(string $name): Kernel
    {
        if (!isset($this->kernels[$name])) {
            throw new GraphException("Unknown kernel: {$name}");
        }

        return $this->kernels[$name];
    }
}
