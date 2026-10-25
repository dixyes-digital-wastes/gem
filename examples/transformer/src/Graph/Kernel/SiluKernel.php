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
use function exp;
use function Toy\Tensor\silu;

final class SiluKernel implements Kernel
{
    public function execute(array $inputs, array $attrs): array|object
    {
        [$input] = $inputs;
        $dimensions = $input->shape()->dimensions();
        if (
            ToyBackend::enabled()
            && $input->dtype() === DType::F32
            && count($dimensions) <= ToyBackend::MAX_RANK
        ) {
            return silu($input->toTensor());
        }
        $data = $input->data();
        $result = [];
        foreach ($data as $value) {
            $result[] = $value / (1.0 + exp(-$value));
        }

        return $result;
    }
}
