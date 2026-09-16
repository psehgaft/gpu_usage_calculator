import asyncio
import shutil


async def sample_nvidia_smi(stop: asyncio.Event) -> list[tuple[float, float]]:
    """Collect aggregate utilization and memory while a local benchmark runs."""
    if not shutil.which("nvidia-smi"):
        return []
    samples: list[tuple[float, float]] = []
    while not stop.is_set():
        process = await asyncio.create_subprocess_exec(
            "nvidia-smi",
            "--query-gpu=utilization.gpu,memory.used",
            "--format=csv,noheader,nounits",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )
        stdout, _ = await process.communicate()
        if process.returncode == 0:
            rows = []
            for line in stdout.decode("utf-8", errors="replace").splitlines():
                try:
                    utilization, memory = (float(value.strip()) for value in line.split(","))
                    rows.append((utilization, memory))
                except (TypeError, ValueError):
                    continue
            if rows:
                samples.append(
                    (
                        sum(row[0] for row in rows) / len(rows),
                        sum(row[1] for row in rows),
                    )
                )
        try:
            await asyncio.wait_for(stop.wait(), timeout=0.5)
        except TimeoutError:
            pass
    return samples
