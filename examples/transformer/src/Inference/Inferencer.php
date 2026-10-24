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

namespace Swow\Examples\Transformer\Inference;

interface Inferencer
{
    /**
     * @return list<float>
     */
    public function forward(int $token, int $position): array;

    public function reset(): void;
}
