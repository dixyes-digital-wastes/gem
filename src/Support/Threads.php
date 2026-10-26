<?php

declare(strict_types=1);

namespace Dixyes\Gem\Support;

use RuntimeException;
use Swow\Thread;

use function array_values;
use function base64_encode;
use function dirname;
use function get_included_files;
use function implode;
use function is_file;
use function is_object;
use function serialize;
use function str_ends_with;
use function str_replace;
use function var_export;

/*
 * Start a worker thread with shared objects.
 *
 * Shared objects (SharedBuffer, Futex, ThreadBarrier) are passed by shared id
 * through serialize, so every worker maps the same memory. Scalars are inlined
 * as literals.
 */
final class Threads
{
    public const TIMEOUT = 300_000;

    /** Locate the Composer autoloader that loaded this process */
    public static function autoloadPath(): string
    {
        foreach (get_included_files() as $file) {
            if (str_ends_with(str_replace('\\', '/', $file), '/vendor/autoload.php')) {
                return $file;
            }
        }

        $local = dirname(__DIR__, 2) . '/vendor/autoload.php';
        if (is_file($local)) {
            return $local;
        }

        throw new RuntimeException('Composer autoloader not found; run composer install');
    }

    /**
     * @param list<object|int|float|string|bool|null> $args
     */
    public static function spawn(string $callable, array $args, ?string $autoload = null): Thread
    {
        $code = 'require ' . var_export($autoload ?? self::autoloadPath(), true) . ';';
        $parts = [];
        foreach (array_values($args) as $index => $arg) {
            if (is_object($arg)) {
                $name = '$shared' . $index;
                $code .= $name . ' = unserialize(base64_decode('
                    . var_export(base64_encode(serialize($arg)), true) . '));';
                $parts[] = $name;
            } else {
                $parts[] = var_export($arg, true);
            }
        }
        $code .= '\\' . $callable . '(' . implode(', ', $parts) . ');';

        return Thread::runCode($code);
    }

    /** @param list<Thread> $workers */
    public static function join(array $workers): void
    {
        foreach ($workers as $id => $worker) {
            $worker->wait(self::TIMEOUT);
            $status = $worker->getExitStatus();
            if ($status !== 0) {
                throw new RuntimeException("Worker {$id} exited with status {$status}");
            }
        }
    }
}
