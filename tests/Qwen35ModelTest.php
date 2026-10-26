<?php

declare(strict_types=1);

namespace Dixyes\Gem\Tests;

use PHPUnit\Framework\TestCase;
use Dixyes\Gem\Gguf\GgufFile;
use Dixyes\Gem\Model\Qwen35Model;
use Dixyes\Gem\Reference\Logits;

use function getenv;
use function is_file;

/**
 * @internal
 */
final class Qwen35ModelTest extends TestCase
{
    private const MODEL = __DIR__ . '/../models/qwen3.5-0.8b/Qwen3.5-0.8B.F32.gguf';

    private const LLAMA_LOGITS = __DIR__ . '/fixtures/llama_logits_00.bin';

    public function testF32LogitsAgainstLlamaCpp(): void
    {
        if (getenv('RUN_QWEN35_SLOW') !== '1') {
            $this->markTestSkipped('Set RUN_QWEN35_SLOW=1 to run the full model');
        }
        if (!is_file(self::MODEL) || !is_file(self::llamaLogits())) {
            $this->markTestSkipped('Qwen3.5 F32 model or llama.cpp logits are not available');
        }

        $file = GgufFile::open(self::MODEL);
        try {
            $model = new Qwen35Model($file);
            $logits = $model->forward(100, 0);
            $expected = Logits::read(self::llamaLogits());
            $result = Logits::compare($expected, $logits);

            $this->assertTrue($result['top1_match']);
            $this->assertTrue($result['top5_match']);
            $this->assertLessThan(1e-4, $result['max_abs']);
        } finally {
            $file->close();
        }
    }

    private static function llamaLogits(): string
    {
        $path = getenv('GEM_LLAMA_LOGITS');

        return $path !== false && $path !== '' ? $path : self::LLAMA_LOGITS;
    }
}
