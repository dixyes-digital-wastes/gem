# gem

PHP matrix kernels and GGUF transformer inference.

The package targets PHP 8.4 or later. Three optional extensions provide the runtime paths:

- `ext-swow` runs inference workers on cooperative threads.
- `ext-toy` accelerates matrix kernels with native code.
- `ext-ffi` binds native libraries for tensor operations.

The worker entry points need a ZTS PHP build with `ext-swow` loaded.

## Install

Install the package with Composer:

```bash
composer require dixyes/gem
```

For a source checkout, install the dependencies:

```bash
composer install
```

The commands in this README use `php bin/...` from a source checkout. An installed project runs the three main entry points from `vendor/bin`.

## Layout

```text
src/        package classes under Dixyes\Gem
bin/        command line entry points
tests/      PHPUnit tests
models/     downloaded GGUF files, ignored by Git
tools/      model download helper
```

## Models

This package uses Qwen3.5-0.8B in Q8_0 or F32 format. Git ignores the model files. Download them with the package tool:

```bash
php tools/download-models.php download --quant=q8
```

The default output directory is `models/qwen3.5-0.8b`. The downloader requires the curl extension. It writes a `.part` file, checks the size and the SHA-256 digest, and resumes an interrupted transfer.

`bin/model.php` offers the same downloader together with metadata inspection:

```bash
php bin/model.php list
php bin/model.php download --source=modelscope --quant=q8
php bin/model.php info models/qwen3.5-0.8b/Qwen3.5-0.8B.Q8_0.gguf --tensors
```

Download options:

```text
--source=hf|modelscope  download source, default hf
--quant=f32|q8|both     quantization to handle, default both
--output=DIR            output directory
--proxy=URL             proxy URL, default HTTPS_PROXY or ALL_PROXY
--token=TOKEN           access token, default HF_TOKEN or MODELSCOPE_API_TOKEN
--force                 replace files that already exist
```

## Generate

```bash
php bin/generate.php models/qwen3.5-0.8b/Qwen3.5-0.8B.Q8_0.gguf --prompt="Hello"
```

The generator defaults to four worker threads. Eight threads decode a few percent faster when eight cores are free, so 8 is the throughput option. Wider pools stay correct but spend more time on barrier latency than they gain; 24 threads only serves the correctness tests.

Options:

```text
--prompt=TEXT       inline prompt
--prompt-file=FILE  read the prompt from a file
--ids=100,101       run raw token ids instead of a prompt
--tokens=N          max generated tokens, default 8; 0 only forwards
--output=PREFIX     save logits as PREFIX_logits_NN.bin
--threads=N         worker threads, default 4
--no-pin            disable CPU pinning
--greedy            greedy sampling
--temp=N --top-k=N --top-p=N --min-p=N --seed=N
```

Use `--ids=... --tokens=0 --output=PREFIX` to dump logits for fixed token ids.

## Chat

```bash
php bin/chat.php models/qwen3.5-0.8b/Qwen3.5-0.8B.Q8_0.gguf
```

`--plain` uses line-based input instead of the terminal interface.

Options:

```text
--context=N       context token limit, default 2048
--tokens=N        max new tokens per turn, default 32
--system=TEXT     system message
--no-system       disable the system message
--thinking        enable the Qwen thinking prompt
--greedy          greedy sampling
--threads=N       inference threads, default 4
--no-pin          disable CPU pinning
--plain           use a line-based prompt instead of the TUI
--temp=N --top-k=N --top-p=N --min-p=N --seed=N
```

## Benchmark

```bash
php bin/bench.php
php bin/bench.php 512 8
```

The benchmark runs the Q8_0 GEMV decode shape and the square GEMM with the selected thread count. `--efficiency` adds a single-thread GEMM pass and reports the speedup.

The matrix kernels also have focused entry points:

```bash
php bin/gemv.php q8_0 both 8
php bin/gemm.php 512 8 --efficiency
php bin/gemv.php f32 --pin=auto
```

`bin/gemv.php` measures `f32`, `q4_0`, and `q8_0` matvec. `bin/gemm.php` measures the square f32 GEMM. Both print the PHP, JIT, and CPU configuration.

## Inspect

```bash
php bin/inspect.php tokenize models/qwen3.5-0.8b/Qwen3.5-0.8B.Q8_0.gguf 'Hello'
php bin/inspect.php sample logits_00.bin --greedy
php bin/inspect.php compare reference.bin actual.bin --top=5
```

`bin/verify.php` checks the quantized kernels against a vector dump:

```bash
php bin/verify.php gemm_vectors.txt
```

## Tests

Run the test suite with PHPUnit:

```bash
composer test
```

Tests that need model files skip when the files are absent. External reference dumps can be supplied through the environment:

```text
GEM_GEMM_VECTORS    vector dump for the graph reference tests
GEM_REFERENCE_DUMP  reference tensor dump for the layer tests
GEM_LLAMA_LOGITS    llama.cpp logits dump for the full model test
RUN_QWEN35_SLOW=1   enable the slow full-model test
```

## License

MIT. See [LICENSE](LICENSE).
