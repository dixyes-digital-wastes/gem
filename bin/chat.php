#!/usr/bin/env php
<?php

declare(strict_types=1);

require __DIR__ . '/bootstrap.php';

use Dixyes\Gem\Chat\ContextLimitException;
use Dixyes\Gem\Chat\Conversation;
use Dixyes\Gem\Gguf\GgufFile;
use Dixyes\Gem\Sampler\Sampler;
use Dixyes\Gem\Sampler\SamplingConfig;
use Dixyes\Gem\Terminal\ChatTui;
use Dixyes\Gem\Terminal\XtermTerminal;
use Dixyes\Gem\Thread\CooperativePool;
use Dixyes\Gem\Thread\ThreadConfig;
use Dixyes\Gem\Tokenizer\QwenTokenizer;

if ($argc < 2) {
    fwrite(STDERR, "usage: chat.php <model.gguf> [options]\n");
    fwrite(STDERR, "  --context=N       context token limit, default 4096\n");
    fwrite(STDERR, "  --tokens=N        max new tokens per turn, default 1024\n");
    fwrite(STDERR, "  --system=TEXT     system message, default \"You are a helpful assistant\"\n");
    fwrite(STDERR, "  --no-system       disable the system message\n");
    fwrite(STDERR, "  --thinking        enable Qwen thinking prompt\n");
    fwrite(STDERR, "  --greedy          greedy sampling\n");
    fwrite(STDERR, "  --threads=N       inference threads, default 4, 8 for throughput\n");
    fwrite(STDERR, "  --no-pin          disable physical-core pinning\n");
    fwrite(STDERR, "  --plain           use a line-based prompt instead of the TUI\n");
    fwrite(STDERR, "  --temp=N --top-k=N --top-p=N --min-p=N --seed=N\n");
    exit(1);
}

$modelPath = $argv[1];
$contextLimit = 4096;
$maxTokens = 1024;
$system = 'You are a helpful assistant';
$thinking = false;
$threads = ThreadConfig::defaultThreads();
$pinned = true;
$plain = false;
$config = new SamplingConfig();
foreach (array_slice($argv, 2) as $argument) {
    if ($argument === '--thinking') {
        $thinking = true;
        continue;
    }
    if ($argument === '--greedy') {
        $config->temperature = 0.0;
        continue;
    }
    if ($argument === '--no-system') {
        $system = null;
        continue;
    }
    if ($argument === '--no-pin') {
        $pinned = false;
        continue;
    }
    if ($argument === '--plain' || $argument === '--no-tui') {
        $plain = true;
        continue;
    }
    if (!str_starts_with($argument, '--') || !str_contains($argument, '=')) {
        fwrite(STDERR, "unknown argument: {$argument}\n");
        exit(1);
    }
    [$key, $value] = explode('=', substr($argument, 2), 2);
    match ($key) {
        'context', 'ctx' => $contextLimit = max(1, (int) $value),
        'tokens', 'max-tokens' => $maxTokens = max(1, (int) $value),
        'system' => $system = $value,
        'threads' => $threads = max(1, (int) $value),
        'temp' => $config->temperature = (float) $value,
        'top-k' => $config->topK = (int) $value,
        'top-p' => $config->topP = (float) $value,
        'min-p' => $config->minP = (float) $value,
        'seed' => $config->seed = (int) $value,
        default => throw new RuntimeException("unknown option: {$key}"),
    };
}

if ($threads > 1 && !class_exists(Swow\Thread::class)) {
    fwrite(STDERR, "warning: Swow Thread is not loaded, using one thread\n");
    $threads = 1;
}
ThreadConfig::setThreads($threads);
ThreadConfig::setPinned($pinned);

ini_set('memory_limit', '-1');
$terminal = $plain ? null : new XtermTerminal();
$file = GgufFile::open($modelPath);
$tokenizer = QwenTokenizer::fromGguf($file);
$pool = new CooperativePool($modelPath, $threads);
$sampler = new Sampler($config);
$conversation = new Conversation(
    $pool,
    $tokenizer,
    $sampler,
    $contextLimit,
    $maxTokens,
    $thinking,
    $system,
);

try {
    if ($plain) {
        if ($system !== null && $system !== '') {
            echo "System: {$system}\n\n";
        }
        while (($line = fgets(STDIN)) !== false) {
            $line = trim($line);
            if ($line === '') {
                continue;
            }
            if ($line === '/exit') {
                break;
            }
            echo "> {$line}\nAssistant: ";
            try {
                $conversation->respond($line, static function (string $piece): void {
                    echo $piece;
                    flush();
                });
            } catch (ContextLimitException) {
                echo '[context limit]';
            }
            echo "\n\n";
        }
    } else {
        (new ChatTui($terminal, $conversation, $system))->run();
    }
} finally {
    $pool->shutdown();
    $file->close();
}
