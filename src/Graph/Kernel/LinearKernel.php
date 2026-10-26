<?php

declare(strict_types=1);

namespace Dixyes\Gem\Graph\Kernel;

use Dixyes\Gem\Graph\DType;
use Dixyes\Gem\Graph\GraphException;
use Dixyes\Gem\Graph\Kernel;
use Dixyes\Gem\Graph\Weight;
use Dixyes\Gem\Thread\Linear;
use Dixyes\Gem\ToyBackend;

use function count;

final class LinearKernel implements Kernel
{
    public function execute(array $inputs, array $attrs): array|object
    {
        [$input] = $inputs;
        $weight = $attrs['weight'] ?? null;
        if (!$weight instanceof Weight) {
            throw new GraphException('linear requires a Weight');
        }
        if ($input->dtype() !== DType::F32) {
            throw new GraphException('linear input must be f32');
        }
        if ($input->shape()->count() !== $weight->columns()) {
            throw new GraphException('linear input width does not match weight columns');
        }
        if (($attrs['transpose'] ?? false) === true) {
            throw new GraphException('transposed linear is not implemented');
        }

        if (!ToyBackend::enabled()) {
            return Linear::matvec($weight, $input->data());
        }

        return Linear::matvec($weight, $input->payload() ?? $input->data(), true);
    }
}
