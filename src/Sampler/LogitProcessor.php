<?php

declare(strict_types=1);

namespace Dixyes\Gem\Sampler;

interface LogitProcessor
{
    /** @param list<float> $logits */
    public function process(array &$logits, SamplingContext $context, SamplingConfig $config): void;
}
