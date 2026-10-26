<?php

declare(strict_types=1);

namespace Dixyes\Gem\Thread;

use Dixyes\Gem\Support\CpuAffinity;
use Dixyes\Gem\Gguf\GgufFile;
use Dixyes\Gem\Model\Qwen35Model;
use Swow\Futex;
use Swow\SharedBuffer;
use Swow\Sync\ThreadBarrier;
use Throwable;

use function file_put_contents;

use const FILE_APPEND;

final class CooperativeWorker
{
    public static function run(
        int $rank,
        int $threads,
        string $ggufPath,
        ?int $cpu,
        SharedBuffer $control,
        SharedBuffer $outputA,
        SharedBuffer $outputB,
        SharedBuffer $logits,
        SharedBuffer $exchange,
        Futex $job,
        CooperativeBarrier $opBarrier,
        ThreadBarrier $tokenBarrier,
    ): void {
        if ($cpu !== null) {
            CpuAffinity::pin($cpu);
        }
        ThreadConfig::setThreads($threads);
        ThreadConfig::setPinned(false);

        $file = GgufFile::open($ggufPath);
        try {
            $model = new Qwen35Model($file);
            CooperativeContext::set(new CooperativeContext(
                $rank,
                $threads,
                $outputA,
                $outputB,
                $logits,
                $exchange,
                $control,
                $opBarrier,
                $tokenBarrier,
            ));

            $sequence = 0;
            while (true) {
                while (($value = $job->getValue()) === $sequence) {
                    $job->wait($value, -1);
                }
                $sequence = $value;
                $command = $control->viewI32(CooperativePool::OFFSET_COMMAND, 1)[0];
                if ($command === CooperativePool::COMMAND_QUIT) {
                    break;
                }

                try {
                    if ($command === CooperativePool::COMMAND_RESET) {
                        $model->reset();
                    } else {
                        $values = $model->forwardTensor(
                            $control->viewI32(CooperativePool::OFFSET_TOKEN, 1)[0],
                            $control->viewI32(CooperativePool::OFFSET_POSITION, 1)[0],
                        );
                        if ($rank === 0) {
                            if ($values->payload() === null) {
                                $logits->writeF32(0, $values->data());
                            } else {
                                $logits->write(0, $values->toBytes());
                            }
                        }
                    }
                } catch (Throwable $e) {
                    $control->writeI32(CooperativePool::OFFSET_ERROR, [1]);
                    file_put_contents(
                        'php://stderr',
                        "cooperative worker {$rank}: {$e}\n",
                        FILE_APPEND,
                    );
                    $tokenBarrier->wait(CooperativePool::BARRIER_TIMEOUT);

                    return;
                }

                if (!$tokenBarrier->wait(CooperativePool::BARRIER_TIMEOUT)) {
                    return;
                }
            }
        } finally {
            CooperativeContext::set(null);
            $file->close();
        }
    }
}
