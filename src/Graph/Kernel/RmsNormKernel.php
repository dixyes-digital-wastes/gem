<?php

declare(strict_types=1);

namespace Dixyes\Gem\Graph\Kernel;

use Dixyes\Gem\Graph\DType;
use Dixyes\Gem\Graph\GraphException;
use Dixyes\Gem\Graph\Kernel;
use Dixyes\Gem\ToyBackend;

use function count;
use function sqrt;
use function Toy\Tensor\rmsnorm;

final class RmsNormKernel implements Kernel
{
    public function execute(array $inputs, array $attrs): array|object
    {
        [$input, $weight] = $inputs;
        $eps = (float) ($attrs['eps'] ?? 1e-6);
        if ($input->dtype() !== DType::F32 || $weight->dtype() !== DType::F32) {
            throw new GraphException('rms_norm requires f32 input and weight');
        }
        if ($input->shape()->count() !== $weight->shape()->count()) {
            throw new GraphException('rms_norm shape mismatch');
        }
        if (
            ToyBackend::enabled()
            && count($input->shape()->dimensions()) === 1
            && count($weight->shape()->dimensions()) === 1
        ) {
            return rmsnorm(
                $input->toTensor(),
                $weight->toTensor(),
                $eps,
            );
        }
        $data = $input->data();
        $weights = $weight->data();
        $square = 0.0;
        foreach ($data as $value) {
            $square += $value * $value;
        }
        $scale = 1.0 / sqrt($square / count($data) + $eps);
        $result = [];
        foreach ($data as $index => $value) {
            $result[] = $value * $scale * $weights[$index];
        }

        return $result;
    }
}
