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

namespace Swow\Examples\Transformer\Terminal;

use RuntimeException;

use function feof;
use function fread;
use function fwrite;
use function ord;
use function shell_exec;
use function substr;
use function trim;

use const STDIN;
use const STDOUT;

final class XtermTerminal
{
    private const ERASE_LINE = "\x1b[2K";

    private const STD_INPUT_HANDLE = -10;
    private const STD_OUTPUT_HANDLE = -11;
    private const ENABLE_PROCESSED_INPUT = 0x0001;
    private const ENABLE_LINE_INPUT = 0x0002;
    private const ENABLE_ECHO_INPUT = 0x0004;
    private const ENABLE_VIRTUAL_TERMINAL_PROCESSING = 0x0004;

    private static ?\FFI $kernel32 = null;

    private ?string $savedModes = null;

    /** @var array{input: int, output: int}|null */
    private ?array $savedWindowsMode = null;

    public function __construct()
    {
        if (!stream_isatty(STDIN) && !self::isWindowsConsole()) {
            throw new RuntimeException('Chat requires a TTY on stdin');
        }
    }

    public function enterRawMode(): void
    {
        if ($this->savedModes !== null || $this->savedWindowsMode !== null) {
            return;
        }
        if (PHP_OS_FAMILY === 'Windows') {
            $this->enterWindowsRawMode();

            return;
        }
        $saved = shell_exec('stty -g < /dev/tty 2>/dev/null');
        if ($saved === null || trim($saved) === '') {
            throw new RuntimeException('Unable to read terminal modes');
        }
        $this->savedModes = trim($saved);
        shell_exec('stty -icanon -echo -isig -ixon < /dev/tty 2>/dev/null');
    }

    public function restore(): void
    {
        if ($this->savedWindowsMode !== null) {
            $this->restoreWindowsMode();

            return;
        }
        if ($this->savedModes === null) {
            return;
        }
        shell_exec('stty ' . $this->savedModes . ' < /dev/tty 2>/dev/null');
        $this->savedModes = null;
        $this->showCursor();
    }

    private static function kernel32(): \FFI
    {
        if (self::$kernel32 === null) {
            self::$kernel32 = \FFI::cdef(<<<'C'
typedef void *HANDLE;
HANDLE GetStdHandle(int nStdHandle);
int GetConsoleMode(HANDLE hConsoleHandle, unsigned long *lpMode);
int SetConsoleMode(HANDLE hConsoleHandle, unsigned long dwMode);
C, 'kernel32.dll');
        }

        return self::$kernel32;
    }

    private static function isWindowsConsole(): bool
    {
        if (PHP_OS_FAMILY !== 'Windows') {
            return false;
        }
        try {
            $ffi = self::kernel32();
            $mode = $ffi->new('unsigned long');

            return $ffi->GetConsoleMode(
                $ffi->GetStdHandle(self::STD_INPUT_HANDLE),
                \FFI::addr($mode),
            ) !== 0;
        } catch (\Throwable) {
            return false;
        }
    }

    private function enterWindowsRawMode(): void
    {
        $ffi = self::kernel32();
        $input = $ffi->GetStdHandle(self::STD_INPUT_HANDLE);
        $output = $ffi->GetStdHandle(self::STD_OUTPUT_HANDLE);
        $inputMode = $ffi->new('unsigned long');
        $outputMode = $ffi->new('unsigned long');
        if ($ffi->GetConsoleMode($input, \FFI::addr($inputMode)) === 0
            || $ffi->GetConsoleMode($output, \FFI::addr($outputMode)) === 0) {
            throw new RuntimeException('Unable to read console modes');
        }

        $newInputMode = $inputMode->cdata & ~(
            self::ENABLE_PROCESSED_INPUT
            | self::ENABLE_LINE_INPUT
            | self::ENABLE_ECHO_INPUT
        );
        if ($ffi->SetConsoleMode($input, $newInputMode) === 0) {
            throw new RuntimeException('Unable to set console input mode');
        }
        $newOutputMode = $outputMode->cdata | self::ENABLE_VIRTUAL_TERMINAL_PROCESSING;
        if ($ffi->SetConsoleMode($output, $newOutputMode) === 0) {
            throw new RuntimeException('Unable to set console output mode');
        }
        $this->savedWindowsMode = [
            'input' => $inputMode->cdata,
            'output' => $outputMode->cdata,
        ];
    }

    private function restoreWindowsMode(): void
    {
        $mode = $this->savedWindowsMode;
        $this->savedWindowsMode = null;
        $this->showCursor();
        if ($mode === null) {
            return;
        }
        $ffi = self::kernel32();
        $ffi->SetConsoleMode($ffi->GetStdHandle(self::STD_INPUT_HANDLE), $mode['input']);
        $ffi->SetConsoleMode($ffi->GetStdHandle(self::STD_OUTPUT_HANDLE), $mode['output']);
    }

    public function __destruct()
    {
        $this->restore();
    }

    /** @return string|false|null string, false on Ctrl-C, null on Ctrl-D */
    public function readLine(string $prompt = ''): string|false|null
    {
        $this->write($prompt);
        $buffer = '';
        while (true) {
            $char = fread(STDIN, 1);
            if ($char === false || $char === '') {
                if (feof(STDIN)) {
                    $this->write("\n");

                    return null;
                }
                continue;
            }
            $byte = ord($char);
            if ($byte === 0x03) {
                $this->write("\r" . self::ERASE_LINE);
                return false;
            }
            if ($byte === 0x04) {
                $this->write("\r" . self::ERASE_LINE . "\n");
                return null;
            }
            if ($byte === 0x1A && PHP_OS_FAMILY === 'Windows') {
                $this->write("\r" . self::ERASE_LINE . "\n");
                return null;
            }
            if ($byte === 0x0A || $byte === 0x0D) {
                $this->write("\n");
                return $buffer;
            }
            if ($byte === 0x7F || $byte === 0x08) {
                $buffer = $this->removeLastCharacter($buffer);
                $this->redraw($prompt, $buffer);
                continue;
            }
            if ($byte === 0x15) {
                $buffer = '';
                $this->redraw($prompt, $buffer);
                continue;
            }
            if ($byte === 0x0C) {
                $this->clearScreen();
                $this->redraw($prompt, $buffer);
                continue;
            }
            if ($byte === 0x1B) {
                $this->consumeEscapeSequence();
                continue;
            }
            if ($byte < 0x20) {
                continue;
            }
            $buffer .= $char;
            $this->write($char);
        }
    }

    public function write(string $text): void
    {
        fwrite(STDOUT, $text);
        fflush(STDOUT);
    }

    public function writeln(string $text = ''): void
    {
        $this->write($text . "\n");
    }

    public function style(string $text, string $sgr): string
    {
        return "\x1b[" . $sgr . 'm' . $text . "\x1b[0m";
    }

    public function bold(string $text): string
    {
        return $this->style($text, '1');
    }

    public function systemLabel(string $text): string
    {
        return $this->style($text, '37;1');
    }

    public function assistantLabel(string $text): string
    {
        return $this->style($text, '36;1');
    }

    public function promptMark(string $text): string
    {
        return $this->style($text, '32;1');
    }

    public function dim(string $text): string
    {
        return $this->style($text, '2');
    }

    public function clearLine(): void
    {
        $this->write("\r" . self::ERASE_LINE);
    }

    public function clearScreen(): void
    {
        $this->write("\x1b[2J\x1b[H");
    }

    public function hideCursor(): void
    {
        $this->write("\x1b[?25l");
    }

    public function showCursor(): void
    {
        $this->write("\x1b[?25h");
    }

    private function redraw(string $prompt, string $buffer): void
    {
        $this->write("\r" . self::ERASE_LINE . $prompt . $buffer);
    }

    private function removeLastCharacter(string $buffer): string
    {
        if ($buffer === '') {
            return '';
        }
        $buffer = substr($buffer, 0, -1);
        while ($buffer !== '' && (ord(substr($buffer, -1)) & 0xC0) === 0x80) {
            $buffer = substr($buffer, 0, -1);
        }
        if ($buffer !== '' && (ord(substr($buffer, -1)) & 0xC0) === 0xC0) {
            $buffer = substr($buffer, 0, -1);
        }

        return $buffer;
    }

    private function consumeEscapeSequence(): void
    {
        $next = fread(STDIN, 1);
        if ($next !== '[') {
            return;
        }
        while (true) {
            $char = fread(STDIN, 1);
            if ($char === false || $char === '') {
                return;
            }
            $byte = ord($char);
            if ($byte >= 0x40 && $byte <= 0x7E) {
                return;
            }
        }
    }
}
