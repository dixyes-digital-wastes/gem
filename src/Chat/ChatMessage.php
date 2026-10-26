<?php

declare(strict_types=1);

namespace Dixyes\Gem\Chat;

final class ChatMessage
{
    public function __construct(
        public string $role,
        public string $content,
        public string $reasoningContent = '',
    ) {
    }
}
