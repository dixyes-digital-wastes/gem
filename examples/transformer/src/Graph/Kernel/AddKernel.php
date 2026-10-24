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

namespace Swow\Examples\Transformer\Graph\Kernel;

use Swow\Examples\Transformer\Graph\Kernel;

final class AddKernel implements Kernel
{
    public function execute(array $inputs, array $attrs): array
    {
        [$left, $right] = $inputs;
        $result = [];
        foreach ($left->data() as $index => $value) {
            $result[] = $value + $right->data()[$index];
        }

        return $result;
    }
}
