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

namespace Swow\Examples\Transformer\Graph;

use Swow\Examples\Transformer\ToyBackend;

use function array_values;
use function count;
use function is_array;
use function pack;

final class Tensor
{
    /** @var list<float>|list<int>|null */
    private ?array $data = null;

    /** @var null|\Toy\Tensor */
    private ?object $payload = null;

    /**
     * @param list<float>|list<int>|\Toy\Tensor $data
     */
    public function __construct(
        private int $id,
        private string $name,
        private Shape $shape,
        private string $dtype,
        array|object $data,
    ) {
        DType::assertSupported($dtype);
        if (is_array($data)) {
            $this->data = array_values($data);
            if (count($this->data) !== $shape->count()) {
                throw new GraphException(
                    "Data count does not match shape {$shape}: " . count($this->data),
                );
            }
        } else {
            $this->payload = $data;
        }
    }

    public function id(): int
    {
        return $this->id;
    }

    public function name(): string
    {
        return $this->name;
    }

    public function shape(): Shape
    {
        return $this->shape;
    }

    public function dtype(): string
    {
        return $this->dtype;
    }

    /** @return list<float>|list<int> */
    public function data(): array
    {
        if ($this->data === null) {
            $this->data = ToyBackend::floats($this->payload);
        }

        return $this->data;
    }

    /** The tensor that already holds this value, null for plain PHP data */
    public function payload(): ?object
    {
        return $this->payload;
    }

    /** The little endian f32 bytes of this value */
    public function toBytes(): string
    {
        if ($this->payload !== null) {
            return $this->payload->toBytes();
        }

        return pack('g*', ...$this->data);
    }

    /**
     * The tensor of this value, converted and cached on first use
     *
     * @return \Toy\Tensor
     */
    public function toTensor(): object
    {
        if ($this->payload === null) {
            $this->payload = ToyBackend::vector($this->data(), $this->shape->dimensions());
        }

        return $this->payload;
    }
}
