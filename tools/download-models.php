#!/usr/bin/env php
<?php

declare(strict_types=1);

const MODEL_NAME = 'Qwen3.5-0.8B';
const MODEL_REPO = 'prithivMLmods/Qwen3.5-0.8B-f32-GGUF';

const MODEL_FILES = [
    'f32' => [
        'file' => 'Qwen3.5-0.8B.F32.gguf',
        'size' => 3_020_533_664,
        'sha256' => '53d5c025769385c3c189cba43be43af53c4c8325dcf29497b44a6cadd8437c23',
    ],
    'q8' => [
        'file' => 'Qwen3.5-0.8B.Q8_0.gguf',
        'size' => 811_843_488,
        'sha256' => '367e9bd0383d7161898c996a58f9a36e6063585d92ae2e5be62e3def0ac1375c',
    ],
];

function usage(int $status): never
{
    $output = $status === 0 ? STDOUT : STDERR;
    fwrite($output, <<<'TXT'
usage: download-models.php <list|download> [options]

commands:
  list      print file URLs, sizes and SHA-256 digests
  download  download the files into --output

options:
  --source=hf|modelscope  download source, default hf
  --quant=f32|q8|both     quantization to handle, default both
  --output=DIR            output directory, default models/qwen3.5-0.8b
  --proxy=URL             proxy URL, default HTTPS_PROXY or ALL_PROXY
  --token=TOKEN           access token, default HF_TOKEN or MODELSCOPE_API_TOKEN
  --force                 replace files that already exist

TXT);
    exit($status);
}

/** @return array{source: string, quant: string, output: string, proxy: ?string, token: ?string, force: bool} */
function parseOptions(array $arguments): array
{
    $options = [
        'source' => 'hf',
        'quant' => 'both',
        'output' => dirname(__DIR__) . '/models/qwen3.5-0.8b',
        'proxy' => null,
        'token' => null,
        'force' => false,
    ];
    $valueOptions = ['source', 'quant', 'output', 'proxy', 'token'];

    foreach ($arguments as $argument) {
        if ($argument === '--force') {
            $options['force'] = true;
            continue;
        }
        if (!str_starts_with($argument, '--') || !str_contains($argument, '=')) {
            throw new InvalidArgumentException("unknown argument: {$argument}");
        }
        [$name, $value] = explode('=', substr($argument, 2), 2);
        if (!in_array($name, $valueOptions, true)) {
            throw new InvalidArgumentException("unknown option: --{$name}");
        }
        $options[$name] = $value;
    }

    if (!in_array($options['source'], ['hf', 'modelscope'], true)) {
        throw new InvalidArgumentException('--source must be hf or modelscope');
    }
    if ($options['quant'] !== 'both' && !array_key_exists($options['quant'], MODEL_FILES)) {
        throw new InvalidArgumentException('--quant must be f32, q8 or both');
    }
    if ($options['output'] === '') {
        throw new InvalidArgumentException('--output must not be empty');
    }

    $options['proxy'] ??= getenv('HTTPS_PROXY') ?: getenv('https_proxy') ?: getenv('ALL_PROXY') ?: getenv('all_proxy') ?: null;
    $options['token'] ??= $options['source'] === 'hf'
        ? (getenv('HF_TOKEN') ?: null)
        : (getenv('MODELSCOPE_API_TOKEN') ?: getenv('MODELSCOPE_TOKEN') ?: null);

    return $options;
}

/** @return list<string> */
function selectedQuants(array $options): array
{
    return $options['quant'] === 'both' ? ['f32', 'q8'] : [$options['quant']];
}

function modelUrl(string $source, string $file): string
{
    return $source === 'hf'
        ? 'https://huggingface.co/' . MODEL_REPO . '/resolve/main/' . $file
        : 'https://modelscope.cn/models/' . MODEL_REPO . '/resolve/master/' . $file;
}

function formatBytes(int $bytes): string
{
    $units = ['B', 'KiB', 'MiB', 'GiB'];
    $value = (float) $bytes;
    foreach ($units as $unit) {
        if ($value < 1024.0 || $unit === 'GiB') {
            return sprintf('%.2f %s', $value, $unit);
        }
        $value /= 1024.0;
    }

    return sprintf('%d B', $bytes);
}

/** @param array{file: string, size: int, sha256: string} $entry */
function verifyFile(string $path, array $entry): bool
{
    if (!is_file($path)) {
        return false;
    }
    clearstatcache(true, $path);
    if (filesize($path) !== $entry['size']) {
        return false;
    }

    fwrite(STDOUT, '  sha256 ... ');
    $digest = hash_file('sha256', $path);
    if ($digest !== $entry['sha256']) {
        fwrite(STDOUT, "mismatch\n");
        return false;
    }
    fwrite(STDOUT, "ok\n");

    return true;
}

/**
 * @param array{file: string, size: int, sha256: string} $entry
 * @param array{token: ?string, proxy: ?string} $options
 */
function transfer(string $url, string $part, int $offset, array $entry, array $options): void
{
    $handle = fopen($part, 'ab');
    if ($handle === false) {
        throw new RuntimeException("cannot open {$part}");
    }

    $headers = ['Accept-Encoding: identity'];
    if ($options['token'] !== null && $options['token'] !== '') {
        $headers[] = 'Authorization: Bearer ' . $options['token'];
    }

    $curl = curl_init($url);
    if ($curl === false) {
        fclose($handle);
        throw new RuntimeException('cannot start curl');
    }

    $started = microtime(true);
    $reported = 0.0;

    curl_setopt_array($curl, [
        CURLOPT_FILE => $handle,
        CURLOPT_FOLLOWLOCATION => true,
        CURLOPT_FAILONERROR => true,
        CURLOPT_CONNECTTIMEOUT => 30,
        CURLOPT_LOW_SPEED_LIMIT => 1024,
        CURLOPT_LOW_SPEED_TIME => 60,
        CURLOPT_HTTPHEADER => $headers,
        CURLOPT_USERAGENT => 'dixyes-gem-model-downloader/0.1',
        CURLOPT_NOPROGRESS => false,
        CURLOPT_XFERINFOFUNCTION => static function (
            CurlHandle $curl,
            int $downloadTotal,
            int $downloaded,
            int $uploadTotal,
            int $uploaded,
        ) use ($offset, $entry, &$started, &$reported): int {
            $now = microtime(true);
            if ($now - $reported < 1.0) {
                return 0;
            }
            $reported = $now;
            $current = min($offset + $downloaded, $entry['size']);
            $rate = $now > $started ? $downloaded / ($now - $started) : 0.0;
            printf(
                "\r  %6.2f%%  %s / %s  %s/s",
                $current * 100.0 / $entry['size'],
                formatBytes($current),
                formatBytes($entry['size']),
                formatBytes((int) $rate),
            );
            fflush(STDOUT);

            return 0;
        },
    ]);
    if ($offset > 0) {
        curl_setopt($curl, CURLOPT_RANGE, $offset . '-');
    }
    if ($options['proxy'] !== null && $options['proxy'] !== '') {
        curl_setopt($curl, CURLOPT_PROXY, $options['proxy']);
    }

    $ok = curl_exec($curl);
    $status = (int) curl_getinfo($curl, CURLINFO_RESPONSE_CODE);
    $error = curl_error($curl);
    unset($curl);
    fclose($handle);
    fwrite(STDOUT, "\n");

    if ($ok === false) {
        throw new RuntimeException("download failed: {$error}");
    }
    if ($status !== 200 && $status !== 206) {
        throw new RuntimeException("unexpected HTTP status {$status}");
    }
}

/** @param array{file: string, size: int, sha256: string} $entry */
function downloadFile(string $source, array $entry, array $options): void
{
    $path = rtrim($options['output'], '/') . '/' . $entry['file'];
    $part = $path . '.part';
    $url = modelUrl($source, $entry['file']);

    if (is_file($path) && !$options['force']) {
        printf("%s: existing file\n", $entry['file']);
        if (verifyFile($path, $entry)) {
            return;
        }
        throw new RuntimeException("{$path}: size or digest mismatch; use --force to replace it");
    }

    if ($options['force']) {
        @unlink($path);
        @unlink($part);
    }

    printf("%s: %s (%s)\n", $entry['file'], formatBytes($entry['size']), $url);

    for ($attempt = 1; $attempt <= 3; $attempt++) {
        clearstatcache(true, $part);
        $offset = is_file($part) ? (int) filesize($part) : 0;

        if ($offset > $entry['size']) {
            throw new RuntimeException("{$part}: partial file is larger than expected");
        }
        if ($offset === $entry['size'] && $offset > 0) {
            if (verifyFile($part, $entry) && rename($part, $path)) {
                printf("  saved %s\n", $path);
                return;
            }
            @unlink($part);
            $offset = 0;
        }

        transfer($url, $part, $offset, $entry, $options);

        clearstatcache(true, $part);
        $written = is_file($part) ? (int) filesize($part) : 0;
        if ($written > $entry['size']) {
            @unlink($part);
            if ($attempt === 3) {
                throw new RuntimeException("{$entry['file']}: server ignored the range request");
            }
            fwrite(STDOUT, "  server ignored the range request, restarting\n");
            continue;
        }
        if ($written === $entry['size']) {
            if (!verifyFile($part, $entry)) {
                @unlink($part);
                throw new RuntimeException("{$entry['file']}: SHA-256 mismatch");
            }
            if (!rename($part, $path)) {
                throw new RuntimeException("cannot rename {$part} to {$path}");
            }
            printf("  saved %s\n", $path);
            return;
        }
        if ($written <= $offset) {
            throw new RuntimeException("{$entry['file']}: download did not make progress");
        }
        printf("  retrying from %s\n", formatBytes($written));
    }

    throw new RuntimeException("{$entry['file']}: download did not complete after 3 attempts");
}

function listModels(array $arguments): void
{
    $options = parseOptions($arguments);

    foreach (selectedQuants($options) as $quant) {
        $entry = MODEL_FILES[$quant];
        printf(
            "%-3s  %10s  %s  %s\n",
            $quant,
            formatBytes($entry['size']),
            $entry['sha256'],
            modelUrl($options['source'], $entry['file']),
        );
    }
}

function downloadModels(array $arguments): void
{
    $options = parseOptions($arguments);
    if (!extension_loaded('curl')) {
        throw new RuntimeException('the curl extension is required for downloads');
    }

    if (!is_dir($options['output'])) {
        if (!mkdir($options['output'], 0o755, true) && !is_dir($options['output'])) {
            throw new RuntimeException("cannot create {$options['output']}");
        }
    }

    printf("%s from %s\n", MODEL_NAME, $options['source']);
    foreach (selectedQuants($options) as $quant) {
        downloadFile($options['source'], MODEL_FILES[$quant], $options);
    }
    printf("done: %s\n", $options['output']);
}

$arguments = array_slice($argv, 1);
$command = array_shift($arguments);

if ($command === null || $command === '--help' || $command === '-h') {
    usage($command === null ? 1 : 0);
}

try {
    match ($command) {
        'list' => listModels($arguments),
        'download' => downloadModels($arguments),
        default => throw new InvalidArgumentException("unknown command: {$command}"),
    };
} catch (Throwable $error) {
    fwrite(STDERR, 'error: ' . $error->getMessage() . "\n");
    exit(1);
}
