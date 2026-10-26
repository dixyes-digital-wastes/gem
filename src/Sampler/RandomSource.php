<?php

declare(strict_types=1);

namespace Dixyes\Gem\Sampler;

interface RandomSource
{
    public function nextUInt32(): int;

    public function nextFloat(): float;
}
