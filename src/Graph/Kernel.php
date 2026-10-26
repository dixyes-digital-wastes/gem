<?php

declare(strict_types=1);

namespace Dixyes\Gem\Graph;

interface Kernel
{
    /**
     * @param list<Tensor> $inputs
     * @param array<string, mixed> $attrs
     * @return list<float>|list<int>|\Toy\Tensor
     */
    public function execute(array $inputs, array $attrs): array|object;
}
