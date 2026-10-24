<?php

declare(strict_types=1);

/*
 * The archive has no examples/autoload.php. The swow repository supplies it.
 * Load the sibling swow checkout's Composer autoloader for the extension
 * classes and library code.
 */
require dirname(__DIR__, 2) . '/swow/vendor/autoload.php';
