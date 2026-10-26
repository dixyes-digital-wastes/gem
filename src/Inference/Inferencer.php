<?php

declare(strict_types=1);

namespace Dixyes\Gem\Inference;

interface Inferencer
{
    /**
     * @return list<float>
     */
    public function forward(int $token, int $position): array;

    public function reset(): void;
}
