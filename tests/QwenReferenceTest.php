<?php

declare(strict_types=1);

namespace Dixyes\Gem\Tests;

use PHPUnit\Framework\TestCase;
use Dixyes\Gem\Gguf\GgufFile;
use Dixyes\Gem\Graph\Graph;
use Dixyes\Gem\Graph\Shape;
use Dixyes\Gem\Graph\Weight;
use Dixyes\Gem\Reference\ReferenceDump;

use function array_values;
use function count;
use function getenv;
use function is_file;
use function unpack;

/**
 * @internal
 */
final class QwenReferenceTest extends TestCase
{
    private const MODEL = __DIR__ . '/../models/qwen3.5-0.8b/Qwen3.5-0.8B.F32.gguf';

    private const REFERENCE = __DIR__ . '/fixtures/ref_tensor_00.bin';

    public function testFfnLayer0AgainstCReference(): void
    {
        $this->assertFfnLayer(0);
    }

    public function testFfnLayer23AgainstCReference(): void
    {
        $this->assertFfnLayer(23);
    }

    public function testPostAttentionNormAgainstCReference(): void
    {
        if (!is_file(self::MODEL) || !is_file(self::referenceDump())) {
            $this->markTestSkipped('Qwen3.5 F32 model or C reference dump is not available');
        }

        $reference = ReferenceDump::open(self::referenceDump());
        $file = GgufFile::open(self::MODEL);
        try {
            $input = $reference->tensor('attn_residual-0')->data();
            $norm = $file->readAll('blk.0.post_attention_norm.weight');
            $normWeight = array_values(unpack('g*', $norm));

            $graph = new Graph();
            $x = $graph->input('attn_residual', new Shape(1024), $input);
            $w = $graph->input('post_attention_norm', new Shape(1024), $normWeight);
            $out = $graph->rmsNorm($x, $w, 1e-6);

            $this->assertClose($reference->tensor('attn_post_norm-0')->data(), $out->data());
        } finally {
            $file->close();
        }
    }

    private function assertFfnLayer(int $layer): void
    {
        if (!is_file(self::MODEL) || !is_file(self::referenceDump())) {
            $this->markTestSkipped('Qwen3.5 F32 model or C reference dump is not available');
        }

        $reference = ReferenceDump::open(self::referenceDump());
        $file = GgufFile::open(self::MODEL);
        try {
            $prefix = 'blk.' . $layer;
            $input = $reference->tensor('attn_post_norm-' . $layer)->data();
            $graph = new Graph();
            $x = $graph->input('attn_post_norm', new Shape(1024), $input);
            $gate = $graph->linear($x, Weight::fromGguf($file, $prefix . '.ffn_gate.weight'));
            $up = $graph->linear($x, Weight::fromGguf($file, $prefix . '.ffn_up.weight'));
            $swiGlu = $graph->swiGlu($gate, $up);
            $out = $graph->linear($swiGlu, Weight::fromGguf($file, $prefix . '.ffn_down.weight'));

            $this->assertClose($reference->tensor('ffn_gate-' . $layer)->data(), $gate->data());
            $this->assertClose($reference->tensor('ffn_up-' . $layer)->data(), $up->data());
            $this->assertClose($reference->tensor('ffn_swiglu-' . $layer)->data(), $swiGlu->data());
            $this->assertClose($reference->tensor('ffn_out-' . $layer)->data(), $out->data());
        } finally {
            $file->close();
        }
    }

    private static function referenceDump(): string
    {
        $path = getenv('GEM_REFERENCE_DUMP');

        return $path !== false && $path !== '' ? $path : self::REFERENCE;
    }

    /** @param list<float> $expected @param list<float> $actual */
    private function assertClose(array $expected, array $actual): void
    {
        $this->assertCount(count($expected), $actual);
        foreach ($expected as $index => $value) {
            $this->assertEqualsWithDelta($value, $actual[$index], max(1e-5, abs($value) * 1e-4));
        }
    }
}
