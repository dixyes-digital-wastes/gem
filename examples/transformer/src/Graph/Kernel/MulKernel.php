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

use Swow\Examples\Transformer\Graph\DType;
use Swow\Examples\Transformer\Graph\Kernel;
use Swow\Examples\Transformer\ToyBackend;

use function count;
use function Toy\Tensor\mul;

final class MulKernel implements Kernel
{
    public function execute(array $inputs, array $attrs): array|object
    {
        [$left, $right] = $inputs;
        $dimensions = $left->shape()->dimensions();
        if (
            ToyBackend::enabled()
            && $left->dtype() === DType::F32
            && $right->dtype() === DType::F32
            && count($dimensions) <= ToyBackend::MAX_RANK
        ) {
            return mul($left->toTensor(), $right->toTensor());
        }
        $leftData = $left->data();
        $rightData = $right->data();
        $result = [];
        foreach ($leftData as $index => $value) {
            $result[] = $value * $rightData[$index];
        }

        return $result;
    }
}
