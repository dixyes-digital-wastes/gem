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

spl_autoload_register(static function (string $class): void {
    $prefix = 'Swow\Examples\Transformer\\';
    if (!str_starts_with($class, $prefix)) {
        return;
    }

    $file = __DIR__ . '/src/' . str_replace('\\', '/', substr($class, strlen($prefix))) . '.php';
    if (is_file($file)) {
        require $file;
    }
});

$gemmAutoload = dirname(__DIR__) . '/gemm/autoload.php';
if (is_file($gemmAutoload)) {
    require $gemmAutoload;
}
