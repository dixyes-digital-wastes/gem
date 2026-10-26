<?php

declare(strict_types=1);

namespace Dixyes\Gem\Graph\Kernel;

use Dixyes\Gem\Graph\DType;
use Dixyes\Gem\Graph\Kernel;
use Dixyes\Gem\ToyBackend;

use function count;
use function Toy\Tensor\add;

final class AddKernel implements Kernel
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
            return add($left->toTensor(), $right->toTensor());
        }
        $leftData = $left->data();
        $rightData = $right->data();
        $result = [];
        foreach ($leftData as $index => $value) {
            $result[] = $value + $rightData[$index];
        }

        return $result;
    }
}
