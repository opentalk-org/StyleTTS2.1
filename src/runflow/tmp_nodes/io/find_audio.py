from pathlib import Path

from runflow.tmp_nodes.io.directory_input import DirectoryInputNode


class FindAudioNode(DirectoryInputNode):
    NODE_TYPE = "FindAudio"
    CATEGORY = "Audio / IO"

    def _load_items(self) -> list[Path]:
        directory = Path(self.params["directory"])
        repeat_count = max(1, int(self.params["repeat_count"]))
        paths = self._audio_paths(directory)
        if not paths:
            paths = [directory / "mock_audio.wav"]
        return [paths[index % len(paths)] for index in range(repeat_count)]

    def _audio_paths(self, directory: Path) -> list[Path]:
        paths: list[Path] = []
        for pattern in self.params["patterns"]:
            paths.extend(sorted(directory.rglob(pattern)))

        seen = set()
        unique = []
        for path in paths:
            resolved = path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            unique.append(path)
        return unique
