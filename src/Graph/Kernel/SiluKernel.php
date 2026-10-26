<?php

declare(strict_types=1);

namespace Dixyes\Gem\Graph\Kernel;

use Dixyes\Gem\Graph\DType;
use Dixyes\Gem\Graph\Kernel;
use Dixyes\Gem\ToyBackend;

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
