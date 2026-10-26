<?php

declare(strict_types=1);

namespace Dixyes\Gem\Thread;

use Swow\Futex;

use function hrtime;

/*
 * Barrier for the cooperative workers.
 *
 * Swow\Sync\ThreadBarrier parks every waiter in a condition variable and
 * measured 129 us per trip for the 4 thread pool, even with no work between
 * the trips. The pool trips an op barrier 97 times per token, so the parking
 * cost alone was 12 ms per token. This barrier publishes a generation and the
 * others observe it while spinning, which measured 0.7 us per trip.
 *
 * The atomic add of the arrival counter picks the last participant: only it
 * resets the counter, publishes the generation and wakes the sleepers. The
 * release sequence of that counter orders the result writes of every
 * participant before the published generation.
 *
 * A participant that misses the generation within the spin budget falls back
 * to the futex wait, so a slow worker costs one wakeup and not a busy core
 */
final class CooperativeBarrier
{
    /* Spin for at most 1 ms. A worker that sleeps pays a wakeup of about
     * 60 us, and a late worker then delays the next trip, so the pool spends
     * the same time in spin and keeps the trips short */
    private const SPIN_NS = 1_000_000;

    private Futex $arrivals;

    private Futex $release;

    private int $count;

    private int $round = 0;

    public function __construct(int $count)
    {
        $this->count = $count;
        $this->arrivals = new Futex(0);
        $this->release = new Futex(0);
    }

    public function wait(int $timeout = -1): bool
    {
        $generation = ++$this->round;
        /* Futex::add() returns the value before the addition, so the last
         * participant is the one that sees count - 1 */
        if ($this->arrivals->add(1) === $this->count - 1) {
            $this->arrivals->setValue(0);
            $this->release->setValue($generation);
            $this->release->wake(-1);

            return true;
        }

        $release = $this->release;
        $deadline = hrtime(true) + self::SPIN_NS;
        while (hrtime(true) < $deadline) {
            if ($release->getValue() === $generation) {
                return true;
            }
        }
        while (true) {
            $value = $release->getValue();
            if ($value === $generation) {
                return true;
            }
            if (!$release->wait($value, $timeout) && $release->getValue() !== $generation) {
                return false;
            }
        }
    }
}
