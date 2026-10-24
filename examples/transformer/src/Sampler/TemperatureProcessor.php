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

namespace Swow\Examples\Transformer\Sampler;

final class TemperatureProcessor implements LogitProcessor
{
    public function process(array &$logits, SamplingContext $context, SamplingConfig $config): void
    {
        if ($config->temperature <= 0.0 || $config->temperature === 1.0) {
            return;
        }
        foreach ($logits as $id => $value) {
            if (is_finite($value)) {
                $logits[$id] = $value / $config->temperature;
            }
        }
    }
}
