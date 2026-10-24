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

namespace Swow\Examples\Transformer\Tokenizer;

use function count;

final class MergeTable
{
    /** @var array<string, int> */
    private array $ranks = [];

    /** @param list<string> $merges */
    public function __construct(array $merges)
    {
        foreach ($merges as $rank => $merge) {
            $parts = explode(' ', trim($merge), 2);
            if (count($parts) !== 2) {
                continue;
            }
            $this->ranks[$parts[0] . "\0" . $parts[1]] = $rank;
        }
    }

    public function rank(string $left, string $right): ?int
    {
        return $this->ranks[$left . "\0" . $right] ?? null;
    }
}
