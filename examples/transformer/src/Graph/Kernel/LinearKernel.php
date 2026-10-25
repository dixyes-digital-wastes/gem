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
use Swow\Examples\Transformer\Graph\GraphException;
use Swow\Examples\Transformer\Graph\Kernel;
use Swow\Examples\Transformer\Graph\Weight;
use Swow\Examples\Transformer\Thread\Linear;
use Swow\Examples\Transformer\ToyBackend;

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
