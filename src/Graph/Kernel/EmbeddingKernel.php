<?php

declare(strict_types=1);

namespace Dixyes\Gem\Graph\Kernel;

use Dixyes\Gem\Quantization\Q4_0;
use Dixyes\Gem\Quantization\Q8_0;
use Dixyes\Gem\Graph\DType;
use Dixyes\Gem\Graph\GraphException;
use Dixyes\Gem\Graph\Kernel;
use Dixyes\Gem\Graph\Weight;

use function array_values;
use function count;
use function unpack;

final class EmbeddingKernel implements Kernel
{
    public function execute(array $inputs, array $attrs): array|object
    {
        [$tokens] = $inputs;
        $weight = $attrs['weight'] ?? null;
        if (!$weight instanceof Weight) {
            throw new GraphException('embedding requires a Weight');
        }
        if ($tokens->dtype() !== DType::I32 || count($tokens->data()) !== 1) {
            throw new GraphException('embedding requires one token id');
        }
        $token = (int) $tokens->data()[0];
        if ($token < 0 || $token >= $weight->rows()) {
            throw new GraphException("Token id out of range: {$token}");
        }
        $row = $weight->row($token);

        return match ($weight->type()) {
            Weight::F32 => array_values(unpack('g*', $row)),
            Weight::Q8_0 => Q8_0::dequantize($row),
            Weight::Q4_0 => Q4_0::dequantize($row),
            default => throw new GraphException('Unsupported weight type: ' . $weight->type()),
        };
    }
}
