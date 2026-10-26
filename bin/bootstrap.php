<?php

declare(strict_types=1);

/*
 * Composer writes the autoloader path into its bin proxy. A direct run from a
 * source checkout has no proxy, so fall back to the local vendor directory.
 */
$autoload = $_composer_autoload_path ?? dirname(__DIR__) . '/vendor/autoload.php';

if (!is_file($autoload)) {
    fwrite(STDERR, "Dependencies are not installed. Run composer install.\n");
    exit(1);
}

require $autoload;
